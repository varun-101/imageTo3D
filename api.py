import os
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from utils import (
    load_image,
    remove_background,
    preprocess_image,
    load_shap_e_models,
    generate_3d_mesh_shap_e,
    save_mesh
)

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Load models once at startup
device, xm, model, diffusion = load_shap_e_models()

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    # Generate a unique ID for this session
    session_id = str(uuid.uuid4())
    input_path = f"input/{session_id}.png"
    nobg_path = f"output/{session_id}_nobg.png"
    obj_path = f"output/{session_id}.obj"

    # Save uploaded image
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Load and process image
    image = load_image(input_path)
    image_nobg = remove_background(image)
    # Save background-removed image
    from PIL import Image
    im = Image.fromarray(image_nobg)
    im.save(nobg_path)

    # Preprocess for model
    image_proc = preprocess_image(image_nobg)
    # Generate mesh
    mesh = generate_3d_mesh_shap_e(
        image_proc, device, xm, model, diffusion
    )
    # Save mesh
    save_mesh(mesh, obj_path)

    return JSONResponse({
        "id": session_id,
        "nobg_url": f"/image/{session_id}_nobg.png",
        "obj_url": f"/model/{session_id}.obj"
    })

@app.get("/image/{filename}")
def get_image(filename: str):
    path = f"output/{filename}"
    if not os.path.exists(path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(path)

@app.get("/model/{filename}")
def get_model(filename: str):
    path = f"output/{filename}"
    if not os.path.exists(path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(path)
