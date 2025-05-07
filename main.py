import argparse
import os
import torch
import numpy as np
from pathlib import Path
from utils import (
    load_image,
    remove_background,
    preprocess_image,
    visualize_3d_model,
    save_mesh,
    load_shap_e_models,
    generate_3d_mesh_shap_e
)

def parse_args():
    parser = argparse.ArgumentParser(description='Convert image to 3D model')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, required=True, help='Path to output 3D model')
    parser.add_argument('--visualize', action='store_true', help='Visualize the 3D model')
    parser.add_argument('--vis_output', type=str, help='Path to save visualization')
    parser.add_argument('--guidance_scale', type=float, default=3.0, help='Guidance scale for Shap-E generation')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load Shap-E models
    print("Loading Shap-E models...")
    device, xm, model, diffusion = load_shap_e_models()
    
    # Load and preprocess image
    print("Loading image...")
    image = load_image(args.input)
    
    print("Removing background...")
    image_no_bg = remove_background(image,args.output)
    
    print("Preprocessing image...")
    image_tensor = preprocess_image(image_no_bg)
    
    # Generate 3D mesh
    print("Generating 3D mesh using Shap-E...")
    mesh = generate_3d_mesh_shap_e(
        image_tensor,
        device,
        xm,
        model,
        diffusion,
        guidance_scale=args.guidance_scale
    )
    
    # Save the mesh
    print(f"Saving mesh to {args.output}...")
    save_mesh(mesh, args.output)
    
    # Visualize if requested
    if args.visualize:
        print("Visualizing 3D model...")
        visualize_3d_model(args.output, args.vis_output)
    
    print("Done!")

if __name__ == "__main__":
    main() 