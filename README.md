# LoRA DreamBooth Training

Hệ thống training và inference LoRA (Low-Rank Adaptation) cho Stable Diffusion với DreamBooth fine-tuning.

## Giới thiệu LoRA

LoRA (Low-Rank Adaptation) là kỹ thuật fine-tuning hiệu quả:
- Thêm các ma trận nhỏ có thể train vào các layer hiện có
- Giữ nguyên base model (tiết kiệm bộ nhớ)
- Tạo file weights nhỏ gọn (1-6 MB thay vì 4-7 GB)
- Cho phép blend linh hoạt giữa base model và fine-tuned behavior

## Tính năng

✅ **DreamBooth Training**: Fine-tune Stable Diffusion trên ảnh của bạn  
✅ **Lightweight**: LoRA weights chỉ 1-6 MB  
✅ **Flexible**: Điều chỉnh mức độ ảnh hưởng từ 0.0 đến 1.0  
✅ **Textual Inversion**: Hỗ trợ custom tokens  
✅ **Demo Scripts**: So sánh trước/sau fine-tuning

## Cài đặt

```bash
# Clone repository
git clone https://github.com/thaihoang104569/gigi
cd gigi

# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt package
pip install -e .
```

## Yêu cầu

- Python 3.8+
- PyTorch 1.13+
- CUDA GPU với 12GB+ VRAM (khuyến nghị)

## Sử dụng

### 1. Chuẩn bị dữ liệu training

Tổ chức ảnh training:
```
my_training_data/
├── image1.jpg
├── image2.jpg
├── image3.jpg
└── ...
```

Khuyến nghị: 5-20 ảnh của đối tượng/style của bạn.

### 2. Train LoRA Model

#### Training cơ bản (UNet only):
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
  --output_format="safe"
```

#### Training nâng cao (UNet + Text Encoder):
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
  --output_format="safe"
```

### 3. Test LoRA (Inference)

#### Demo script nhanh:
```bash
python demo_inference.py \
  --lora_path="output/my_lora/lora_weight.safetensors" \
  --prompt="a photo of sks person" \
  --alphas 0.0 0.5 1.0
```

Kết quả:
- `output/alpha_0.00.png` - Base model (trước fine-tuning)
- `output/alpha_0.50.png` - 50% LoRA
- `output/alpha_1.00.png` - Full LoRA (sau fine-tuning)
- `output/comparison_grid.png` - Grid so sánh

#### Code Python:
```python
import torch
from diffusers import StableDiffusionPipeline
from lora_diffusion import patch_pipe, tune_lora_scale

# Load Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Generate ảnh TRƯỚC fine-tuning
image_before = pipe("a photo of sks person", num_inference_steps=50).images[0]
image_before.save("before.png")

# Apply LoRA weights
patch_pipe(
    pipe,
    "output/my_lora/lora_weight.safetensors",
    patch_text=True,
    patch_ti=True,
    patch_unet=True
)

# Set LoRA strength (0.0 = tắt, 1.0 = full)
tune_lora_scale(pipe.unet, 1.0)
tune_lora_scale(pipe.text_encoder, 1.0)

# Generate ảnh SAU fine-tuning
image_after = pipe("a photo of sks person", num_inference_steps=50).images[0]
image_after.save("after.png")
```

## Tham số Training quan trọng

| Tham số | Mô tả | Khuyến nghị |
|---------|-------|-------------|
| `--lora_rank` | Rank của ma trận LoRA (cao hơn = capacity cao hơn) | 4, 8, hoặc 16 |
| `--learning_rate` | Learning rate cho UNet | 1e-4 |
| `--learning_rate_text` | Learning rate cho text encoder | 5e-5 |
| `--max_train_steps` | Tổng số bước training | 2000-5000 |
| `--train_text_encoder` | Fine-tune cả text encoder | Khuyến nghị |
| `--gradient_checkpointing` | Giảm memory usage | Cho VRAM hạn chế |
| `--use_8bit_adam` | Dùng 8-bit Adam optimizer | Cho VRAM hạn chế |

## Tính năng nâng cao

### Điều chỉnh LoRA Strength
```python
# Test các mức độ ảnh hưởng khác nhau
for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
    tune_lora_scale(pipe.unet, alpha)
    tune_lora_scale(pipe.text_encoder, alpha)
    image = pipe(prompt).images[0]
    image.save(f"alpha_{alpha}.png")
```

### Merge nhiều LoRAs
```bash
lora_add \
  path/to/lora1.safetensors \
  path/to/lora2.safetensors \
  0.5 0.5 \
  output/merged_lora.safetensors
```

### Textual Inversion Support

LoRA files có thể bao gồm custom tokens (VD: `<s1><s2>`, `<sks>`):
```python
prompt = "a portrait of <s1><s2> in cyberpunk style"
```

Tokens được tự động load khi dùng `patch_pipe()`.

## Tối ưu bộ nhớ

Với GPU VRAM hạn chế:

```bash
python training_scripts/train_lora_dreambooth.py \
  --gradient_checkpointing \
  --use_8bit_adam \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  ...
```

Cho inference:
```python
# Dùng half precision
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")
```

## Cấu trúc Project

```
lora/
├── demo_inference.py              # Demo inference script
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
├── lora_diffusion/                # Core library
│   ├── lora.py                    # LoRA implementation
│   ├── utils.py                   # Utilities
│   ├── lora_manager.py            # Multi-LoRA management
│   └── xformers_utils.py          # Memory optimization
└── training_scripts/              # Training scripts
    └── train_lora_dreambooth.py   # Main DreamBooth trainer
```

## Xử lý lỗi

**Q: Out of memory khi training?**  
A: Dùng `--gradient_checkpointing`, `--use_8bit_adam`, giảm `--train_batch_size` xuống 1

**Q: Training quá chậm?**  
A: Cài `xformers`: `pip install xformers`, thêm flag `--use_xformers`

**Q: LoRA effect quá yếu/mạnh?**  
A: Điều chỉnh bằng `tune_lora_scale()` hoặc tăng/giảm `--lora_rank`

**Q: Ảnh không giống training data?**  
A: Train lâu hơn (`--max_train_steps`), dùng `--train_text_encoder`, hoặc tăng `--lora_rank`

## License

MIT License

## Author

thaihoang104569
