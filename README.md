# LoRA Diffusion - DreamBooth Training & Inference

Low-Rank Adaptation (LoRA) for Stable Diffusion with DreamBooth fine-tuning. This simplified version focuses on training custom LoRA models and generating images before/after fine-tuning.

## What is LoRA?

LoRA (Low-Rank Adaptation) is an efficient fine-tuning technique that:
- Adds small trainable matrices to existing model layers
- Keeps the base model frozen (saves memory)
- Produces tiny weight files (1-6 MB vs 4-7 GB for full models)
- Allows dynamic blending between base and fine-tuned behavior

## Features

✅ **DreamBooth Training**: Fine-tune Stable Diffusion on your own images  
✅ **Lightweight**: LoRA weights are only 1-6 MB  
✅ **Flexible Inference**: Adjust LoRA influence from 0.0 (base) to 1.0 (full)  
✅ **Textual Inversion**: Support for custom tokens like `<s1><s2>`  
✅ **Easy Demo**: Simple scripts to compare before/after fine-tuning

## Installation

```bash
# Clone the repository
git clone https://github.com/cloneofsimo/lora
cd lora

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Requirements

- Python 3.8+
- PyTorch 1.13+
- CUDA GPU with 12GB+ VRAM (recommended)
- See `requirements.txt` for full dependencies

## Quick Start

### 1. Prepare Your Training Data

Organize your training images:
```
my_training_data/
├── image1.jpg
├── image2.jpg
├── image3.jpg
└── ...
```

Recommended: 5-20 images of your subject/style.

### 2. Train a LoRA Model (DreamBooth)

#### Basic Training (UNet only):
```bash
python training_scripts/train_lora_dreambooth.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="my_training_data" \
  --output_dir="output/my_lora" \
  --instance_prompt="a photo of sks person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=3000 \
  --lora_rank=4 \
  --output_format="safetensors"
```

#### Advanced Training (UNet + Text Encoder):
```bash
python training_scripts/train_lora_dreambooth.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="my_training_data" \
  --output_dir="output/my_lora" \
  --instance_prompt="a photo of sks person" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --learning_rate_text=5e-5 \
  --train_text_encoder \
  --color_jitter \
  --max_train_steps=2000 \
  --lora_rank=4 \
  --output_format="safetensors"
```

See `training_scripts/run_lora_db_unet_only.sh` and `training_scripts/run_lora_db_w_text.sh` for complete examples.

### 3. Test Your LoRA (Inference Demo)

#### Quick Demo Script:
```bash
python demo_inference.py \
  --lora_path="output/my_lora/lora.safetensors" \
  --prompt="a photo of sks person" \
  --alphas 0.0 0.5 1.0
```

This generates:
- `output/alpha_0.00.png` - Base model (before fine-tuning)
- `output/alpha_0.50.png` - 50% LoRA influence
- `output/alpha_1.00.png` - Full LoRA effect (after fine-tuning)
- `output/comparison_grid.png` - Side-by-side comparison

#### Python Code Example:
```python
import torch
from diffusers import StableDiffusionPipeline
from lora_diffusion import patch_pipe, tune_lora_scale

# Load base Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Generate image BEFORE fine-tuning
image_before = pipe("a photo of sks person", num_inference_steps=50).images[0]
image_before.save("before.png")

# Apply LoRA weights
patch_pipe(
    pipe,
    "output/my_lora/lora.safetensors",
    patch_text=True,
    patch_ti=True,
    patch_unet=True
)

# Set LoRA strength (0.0 = off, 1.0 = full)
tune_lora_scale(pipe.unet, 1.0)
tune_lora_scale(pipe.text_encoder, 1.0)

# Generate image AFTER fine-tuning
image_after = pipe("a photo of sks person", num_inference_steps=50).images[0]
image_after.save("after.png")
```

### 4. Jupyter Notebooks

Interactive demos available in `scripts/`:
- **`run_inference.ipynb`** - Text-to-image with LoRA
- **`run_img2img.ipynb`** - Image-to-image with LoRA

### 5. Example LoRAs

Pre-trained LoRAs for testing in `example_loras/`:
- `lora_disney.safetensors` - Disney animation style
- `lora_popart.safetensors` - Pop art style
- `lora_illust.safetensors` - Illustration style

Try them:
```bash
python demo_inference.py \
  --lora_path="example_loras/lora_disney.safetensors" \
  --prompt="a cute baby lion in the style of <s1><s2>"
```

## Key Training Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--lora_rank` | LoRA matrix rank (higher = more capacity) | 4, 8, or 16 |
| `--learning_rate` | UNet learning rate | 1e-4 |
| `--learning_rate_text` | Text encoder learning rate | 5e-5 |
| `--max_train_steps` | Total training steps | 2000-5000 |
| `--train_text_encoder` | Also fine-tune text encoder | Recommended |
| `--gradient_checkpointing` | Reduce memory usage | For limited VRAM |
| `--use_8bit_adam` | Use 8-bit Adam optimizer | For limited VRAM |

## Advanced Features

### Adjust LoRA Strength Dynamically
```python
# Test different influence levels
for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
    tune_lora_scale(pipe.unet, alpha)
    tune_lora_scale(pipe.text_encoder, alpha)
    image = pipe(prompt).images[0]
    image.save(f"alpha_{alpha}.png")
```

### Merge Multiple LoRAs
```bash
lora_add \
  path/to/lora1.safetensors \
  path/to/lora2.safetensors \
  0.5 0.5 \
  output/merged_lora.safetensors
```

### Textual Inversion Support

LoRA files can include custom tokens (e.g., `<s1><s2>`, `<sks>`):
```python
prompt = "a portrait of <s1><s2> in cyberpunk style"
```

The tokens are automatically loaded when using `patch_pipe()`.

## Memory Optimization

For GPUs with limited VRAM:

```bash
python training_scripts/train_lora_dreambooth.py \
  --gradient_checkpointing \
  --use_8bit_adam \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  ...
```

For inference:
```python
# Use half precision
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")
```

## Project Structure

```
lora/
├── demo_inference.py              # Quick inference demo script
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
├── lora_diffusion/                # Core library
│   ├── lora.py                    # LoRA implementation
│   ├── utils.py                   # Image grid, utilities
│   ├── lora_manager.py            # Multi-LoRA management
│   ├── xformers_utils.py          # Memory optimization
│   └── cli_lora_add.py            # LoRA merging CLI
├── training_scripts/              # Training scripts
│   ├── train_lora_dreambooth.py   # Main DreamBooth trainer
│   ├── run_lora_db_unet_only.sh   # Example: UNet only
│   └── run_lora_db_w_text.sh      # Example: UNet + Text
├── scripts/                       # Jupyter notebooks
│   ├── run_inference.ipynb        # Text-to-image demo
│   └── run_img2img.ipynb          # Img2img demo
└── example_loras/                 # Pre-trained examples
    ├── lora_disney.safetensors
    ├── lora_popart.safetensors
    └── lora_illust.safetensors
```

## Troubleshooting

**Q: Out of memory error during training?**  
A: Use `--gradient_checkpointing`, `--use_8bit_adam`, reduce `--train_batch_size` to 1

**Q: Training is too slow?**  
A: Install `xformers`: `pip install xformers`, add `--use_xformers` flag

**Q: LoRA effect is too weak/strong?**  
A: Adjust with `tune_lora_scale()` or increase/decrease `--lora_rank`

**Q: Images don't match training data?**  
A: Train longer (`--max_train_steps`), use `--train_text_encoder`, or increase `--lora_rank`

## Citation

If you use this code, please cite:

```bibtex
@misc{ryu2023lora,
  author = {Simo Ryu},
  title = {Low-Rank Adaptation for Fast Text-to-Image Diffusion Fine-tuning},
  year = {2023},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/cloneofsimo/lora}}
}
```

## License

MIT License - See LICENSE file for details

## Resources

- **Original Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **DreamBooth Paper**: [DreamBooth: Fine Tuning Text-to-Image Diffusion Models](https://arxiv.org/abs/2208.12242)
- **Stable Diffusion**: [Hugging Face Diffusers](https://github.com/huggingface/diffusers)

## Credits

Original implementation by [Simo Ryu](https://github.com/cloneofsimo). This simplified version focuses on DreamBooth training and inference workflows.
