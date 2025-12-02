"""
Simple demo script for comparing Stable Diffusion image generation 
before and after applying LoRA fine-tuning.

Usage:
    python demo_inference.py --lora_path example_loras/lora_disney.safetensors
"""

import argparse
import torch
from diffusers import StableDiffusionPipeline
from lora_diffusion import patch_pipe, tune_lora_scale, image_grid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Demo: Compare base SD vs LoRA fine-tuned SD")
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Pretrained Stable Diffusion model ID"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="example_loras/lora_disney.safetensors",
        help="Path to LoRA weights (.pt or .safetensors)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a cute baby lion in the style of <s1><s2>",
        help="Text prompt for image generation"
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.3, 0.5, 0.7, 1.0],
        help="List of LoRA alpha values to test (0.0 = base model, 1.0 = full LoRA)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for classifier-free guidance"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading base model: {args.model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        safety_checker=None
    ).to(args.device)
    
    print(f"Applying LoRA weights from: {args.lora_path}")
    patch_pipe(
        pipe,
        args.lora_path,
        patch_text=True,
        patch_ti=True,
        patch_unet=True,
    )
    
    images = []
    print(f"\nGenerating images with prompt: '{args.prompt}'")
    print("=" * 60)
    
    for alpha in args.alphas:
        print(f"Alpha = {alpha:.2f} ", end="")
        
        # Adjust LoRA scale
        tune_lora_scale(pipe.unet, alpha)
        tune_lora_scale(pipe.text_encoder, alpha)
        
        # Generate image
        torch.manual_seed(args.seed)
        with torch.no_grad():
            image = pipe(
                args.prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale
            ).images[0]
        
        images.append(image)
        
        # Save individual image
        output_path = output_dir / f"alpha_{alpha:.2f}.png"
        image.save(output_path)
        print(f"✓ Saved to {output_path}")
    
    # Create comparison grid
    print("\nCreating comparison grid...")
    grid = image_grid(images, rows=1, cols=len(args.alphas))
    grid_path = output_dir / "comparison_grid.png"
    grid.save(grid_path)
    print(f"✓ Comparison grid saved to {grid_path}")
    
    print("\n" + "=" * 60)
    print("Done! Results:")
    print(f"  - Individual images: {output_dir}/alpha_*.png")
    print(f"  - Comparison grid: {grid_path}")
    print(f"\nInterpretation:")
    print(f"  Alpha = 0.0: Base Stable Diffusion (before fine-tuning)")
    print(f"  Alpha = 1.0: Full LoRA effect (after fine-tuning)")
    print(f"  Alpha in between: Interpolation between base and fine-tuned")


if __name__ == "__main__":
    main()
