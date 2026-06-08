# D:\Projects\Friendship\backend\main.py
import io
import json
import numpy as np
from PIL import Image
from pathlib import Path
import uuid
from datetime import datetime

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ========== CONFIG ==========
THRESHOLD = 0.00 
BACKEND_DIR = Path(__file__).parent
MASTER_EMB_PATH = BACKEND_DIR / "embeddings_output" / "master_embedding.json"
SAVED_FACES_DIR = BACKEND_DIR / "saved_faces"

# Create saved faces directory if it doesn't exist
SAVED_FACES_DIR.mkdir(exist_ok=True)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# ========== LOAD MODELS ==========
print("📦 Loading MTCNN face detector...")
mtcnn = MTCNN(
    image_size=160, 
    margin=20, 
    keep_all=True, 
    thresholds=[0.5, 0.6, 0.6],  # Sensitive thresholds for better detection
    device=device
)

print("🤖 Loading FaceNet model...")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("✅ Models loaded successfully!")

# ========== LOAD MASTER EMBEDDING ==========
master_emb = None
if MASTER_EMB_PATH.exists():
    print("📂 Loading master embedding...")
    with open(MASTER_EMB_PATH, "r") as f:
        master_data = json.load(f)
    master_emb = np.array(master_data["master_embedding"], dtype=np.float32)
    master_emb = master_emb / np.linalg.norm(master_emb)
    print("✅ Master embedding loaded")
else:
    print("⚠️  No master embedding found. Please register a face first.")

# ========== HELPER FUNCTIONS ==========
def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_largest_face(faces, boxes):
    """Get the largest face from detected faces"""
    if faces is None or len(faces) == 0:
        return None
    
    # Calculate face areas from bounding boxes
    areas = []
    for box in boxes:
        area = (box[2] - box[0]) * (box[3] - box[1])
        areas.append(area)
    
    # Get index of largest face
    largest_idx = np.argmax(areas)
    
    # Return the largest face
    if faces.dim() == 4:
        return faces[largest_idx]
    else:
        return faces

def extract_embedding(image):
    """Extract face embedding from PIL image"""
    # Detect faces with sensitive settings
    boxes, probs = mtcnn.detect(image)
    
    if boxes is None:
        return None
    
    # Get the face tensor
    faces = mtcnn(image)
    
    if faces is None:
        return None
    
    # Get the largest face
    face = get_largest_face(faces, boxes)
    
    if face is None:
        return None
    
    if face.dim() == 3:
        face = face.unsqueeze(0)
    
    face = face.to(device)
    
    with torch.no_grad():
        embedding = model(face).cpu().numpy()[0]
    
    # Normalize embedding
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding

def save_photo(img_bytes, verified, similarity):
    """Save captured photo to disk"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    status = "verified" if verified else "rejected"
    filename = SAVED_FACES_DIR / f"{timestamp}_{unique_id}_{status}_sim{similarity:.2f}.jpg"
    
    with open(filename, "wb") as f:
        f.write(img_bytes)
    
    print(f"📸 Photo saved: {filename.name}")
    return filename

# ========== FASTAPI APP ==========
app = FastAPI(
    title="Face Verification API",
    description="Simple face verification system",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== ENDPOINTS ==========
@app.get("/")
def home():
    """Health check endpoint"""
    return {
        "status": "online",
        "device": device,
        "master_registered": master_emb is not None,
        "threshold": THRESHOLD
    }

@app.post("/register")
async def register_face(file: UploadFile = File(...)):
    """Register a new master face from uploaded photo"""
    global master_emb
    
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Save registration photo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = SAVED_FACES_DIR / f"registration_{timestamp}_{unique_id}.jpg"
        with open(filename, "wb") as f:
            f.write(img_bytes)
        print(f"📸 Registration photo saved: {filename.name}")
        
        embedding = extract_embedding(img)
        
        if embedding is None:
            return {
                "success": False,
                "error": "No face detected in the image. Please upload a clear face photo."
            }
        
        master_emb = embedding
        
        # Save to file
        MASTER_EMB_PATH.parent.mkdir(exist_ok=True)
        with open(MASTER_EMB_PATH, "w") as f:
            json.dump({"master_embedding": master_emb.tolist()}, f)
        
        return {
            "success": True,
            "message": "Master face registered successfully!",
            "embedding_dimension": len(master_emb)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/verify")
async def verify_face(file: UploadFile = File(...)):
    """Verify if uploaded face matches the registered master face"""
    global master_emb
    
    if master_emb is None:
        return {
            "verified": False,
            "error": "No master face registered. Please register first."
        }
    
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        embedding = extract_embedding(img)
        
        if embedding is None:
            # Save photo with no face detected
            save_photo(img_bytes, False, 0.0)
            return {
                "verified": False,
                "error": "No face detected in the uploaded image"
            }
        
        similarity = cosine_similarity(embedding, master_emb)
        verified = similarity >= THRESHOLD
        
        # Save the captured photo
        save_photo(img_bytes, verified, similarity)
        
        return {
            "verified": verified,
            "similarity": round(similarity, 4),
            "threshold": THRESHOLD,
            "confidence": "High" if similarity > 0.5 else "Medium" if similarity > 0.07 else "Low"
        }
        
    except Exception as e:
        return {"verified": False, "error": str(e)}

@app.post("/verify-from-multiple")
async def verify_multiple(files: list[UploadFile] = File(...)):
    """Verify multiple faces against master (returns best match)"""
    global master_emb
    
    if master_emb is None:
        return {"error": "No master face registered"}
    
    results = []
    
    for file in files:
        try:
            img_bytes = await file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            embedding = extract_embedding(img)
            
            if embedding is not None:
                similarity = cosine_similarity(embedding, master_emb)
                results.append({
                    "filename": file.filename,
                    "similarity": round(similarity, 4),
                    "verified": similarity >= THRESHOLD
                })
                # Save multi-verify photos
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = SAVED_FACES_DIR / f"multiverify_{timestamp}_{file.filename}"
                with open(filename, "wb") as f:
                    f.write(img_bytes)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    if not results:
        return {"error": "No valid faces detected in any image"}
    
    best = max(results, key=lambda x: x.get("similarity", 0))
    
    return {
        "best_match": best,
        "all_results": results,
        "threshold": THRESHOLD
    }

@app.delete("/reset")
async def reset_master():
    """Reset/delete the master embedding"""
    global master_emb
    
    master_emb = None
    if MASTER_EMB_PATH.exists():
        MASTER_EMB_PATH.unlink()
    
    return {"success": True, "message": "Master face reset successfully"}

@app.get("/saved-photos")
async def list_saved_photos():
    """List all saved photos (for debugging)"""
    photos = []
    for img_path in SAVED_FACES_DIR.glob("*.jpg"):
        photos.append({
            "filename": img_path.name,
            "size_kb": round(img_path.stat().st_size / 1024, 2),
            "modified": datetime.fromtimestamp(img_path.stat().st_mtime).isoformat()
        })
    return {"photos": photos, "count": len(photos)}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 Starting Face Verification API")
    print("="*50)
    print(f"📱 API URL: http://localhost:8000")
    print(f"📚 Documentation: http://localhost:8000/docs")
    print(f"📸 Saved photos will be stored in: {SAVED_FACES_DIR}")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)