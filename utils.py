import cv2
import numpy as np
from rembg import remove
from PIL import Image
import trimesh
import pyrender
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh
from shap_e.rendering.torch_mesh import TriMesh
def load_shap_e_models():
    """
    Load the Shap-E models required for 3D generation.
    
    Returns:
        tuple: (device, xm, model, diffusion) needed for generation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the transmitter model for converting latents to 3D
    xm = load_model('transmitter', device=device)
    
    # Load the image model
    model = load_model('image300M', device=device)
    
    # Load the diffusion model
    diffusion = diffusion_from_config(load_config('diffusion'))
    
    return device, xm, model, diffusion

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from the given path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        np.ndarray: Loaded image in BGR format
    """
    # Read image using cv2
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return img

import os
import uuid
import numpy as np
import cv2
from PIL import Image
from rembg import remove

def remove_background(image: np.ndarray, output_dir: str = "output") -> np.ndarray:
    """
    Remove background from the input image and save the result in the output folder.
    
    Args:
        image (np.ndarray): Input image in BGR format
        output_dir (str): Directory to save the output image
        
    Returns:
        np.ndarray: Image with background removed (RGBA)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)
    # Remove background
    output = remove(pil_image)
    # Generate a unique filename
    filename = f"{uuid.uuid4().hex}.png"
    save_path = os.path.join(output_dir, filename)
    # Save the image
    output.save(save_path)
    print(f"Saved background-removed image to: {save_path}")
    # Convert back to numpy array and return
    return np.array(output)


def preprocess_image(image: np.ndarray, target_size: tuple = (256, 256)) -> Image.Image:
    """
    Preprocess image for Shap-E model.
    
    Args:
        image (np.ndarray): Input image
        target_size (tuple): Target size for resizing
        
    Returns:
        PIL.Image: Preprocessed image
    """
    # Convert to PIL Image if not already
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Resize image
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    return image

def generate_3d_mesh_shap_e(image: Image.Image, device: torch.device, xm, model, diffusion, guidance_scale: float = 3.0):
    """
    Generate a 3D mesh using Shap-E from the preprocessed image.
    
    Args:
        image (PIL.Image): Preprocessed image
        device (torch.device): Device to run the model on
        xm: Transmitter model for converting latents to 3D
        model: Image model for generating latents
        diffusion: Diffusion model
        guidance_scale (float): Guidance scale for generation
        
    Returns:
        trimesh.Trimesh: Generated 3D mesh
    """
    batch_size = 1  # We only generate one mesh at a time
    
    # Sample latents using the official method
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(images=[image] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    
    # Convert the first latent to a mesh
    mesh = decode_latent_mesh(xm, latents[0])
    
    # Get the vertices and faces
    vertices = mesh.verts.cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    
    # Create a trimesh object
    # tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # # Calculate vertex normals and use them for colors
    # vertex_normals = tri_mesh.vertex_normals
    # vertex_colors = (vertex_normals + 1) / 2  # Convert from [-1,1] to [0,1] range
    
    # Update the mesh with vertex colors
    # tri_mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def visualize_3d_model(mesh_path: str, save_path: str = None):
    """
    Visualize a 3D model using pyrender.
    
    Args:
        mesh_path (str): Path to the 3D model file
        save_path (str, optional): Path to save the visualization
    """
    # Load the mesh
    mesh = trimesh.load(mesh_path)
    
    # Convert trimesh to pyrender mesh
    mesh = pyrender.Mesh.from_trimesh(mesh)
    
    # Create a scene and add the mesh
    scene = pyrender.Scene()
    scene.add(mesh)
    
    # Add lighting
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light)
    
    # Create a camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    s = np.sqrt(2)/2
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, s, -s, -2.0],
        [0.0, s, s, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)
    
    # Render the scene
    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
    color, _ = r.render(scene)
    
    # Display or save the visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(color)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def save_mesh(mesh: trimesh.Trimesh, output_path: str):
    """
    Save the generated mesh to a file.
    
    Args:
        mesh (trimesh.Trimesh): Mesh to save
        output_path (str): Path to save the mesh
    """
    # Get the file extension
    ext = Path(output_path).suffix.lower()
    
    # Save in the appropriate format
    if ext == '.obj':
        mesh.export(output_path, file_type='obj')
    elif ext == '.stl':
        mesh.export(output_path, file_type='stl')
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .obj or .stl") 