import argparse
import os
from pathlib import Path
from PIL import Image, ImageEnhance
from tqdm import tqdm


def resize_image(image, target_size=512, keep_aspect_ratio=True):
    """
    Resize ảnh về target_size, giữ nguyên tỷ lệ và pad nếu cần
    """
    width, height = image.size
    
    if keep_aspect_ratio:
        # Resize giữ tỷ lệ, pad phần còn lại
        ratio = min(target_size / width, target_size / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # Resize
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Tạo canvas vuông với background trắng
        canvas = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        
        # Paste vào giữa
        offset_x = (target_size - new_width) // 2
        offset_y = (target_size - new_height) // 2
        canvas.paste(resized, (offset_x, offset_y))
        
        return canvas
    else:
        # Resize trực tiếp (có thể bị méo)
        return image.resize((target_size, target_size), Image.LANCZOS)


def enhance_image(image, brightness=1.1, contrast=1.1, sharpness=1.1):
    """
    Enhance brightness, contrast, sharpness
    """
    # Brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    
    # Contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    
    # Sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness)
    
    return image


def preprocess_image(image_path, output_path, target_size=512, enhance=True, keep_aspect_ratio=True):
    """
    Xử lý 1 ảnh: load -> resize -> enhance -> save
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize (giữ nguyên tỷ lệ hoặc không)
        image = resize_image(image, target_size, keep_aspect_ratio)
        
        # Enhance quality
        if enhance:
            image = enhance_image(image)
        
        # Save
        image.save(output_path, quality=95)
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def preprocess_folder(input_dir, output_dir, target_size=512, enhance=True, keep_aspect_ratio=True):
    """
    Xử lý toàn bộ folder ảnh
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jfif', '.JPG', '.JPEG', '.PNG', '.JFIF'}
    
    # Get all images
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix in image_extensions]
    
    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Target size: {target_size}x{target_size}")
    print(f"Keep aspect ratio: {'YES (with padding)' if keep_aspect_ratio else 'NO (stretch)'}")
    print(f"Enhancement: {'ON' if enhance else 'OFF'}")
    print(f"Output: {output_dir}\n")
    
    success_count = 0
    
    # Process each image
    for img_file in tqdm(image_files, desc="Processing images"):
        output_file = output_path / f"{img_file.stem}_processed{img_file.suffix}"
        
        if preprocess_image(img_file, output_file, target_size, enhance, keep_aspect_ratio):
            success_count += 1
    
    print(f"\nSuccessfully processed: {success_count}/{len(image_files)} images")
    print(f"Output folder: {output_dir}")
    
    # Create caption file (optional)
    caption_file = output_path / "captions.txt"
    with open(caption_file, 'w', encoding='utf-8') as f:
        f.write("# Sử dụng prompt này cho training:\n")
        f.write("# --instance_prompt=\"a photo of sks person\"\n")
        f.write("# hoặc thay 'sks' bằng tên bạn muốn\n")
    
    print(f"Created caption guide: {caption_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Tiền xử lý ảnh cho DreamBooth LoRA training"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder chứa ảnh gốc"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Folder lưu ảnh đã xử lý"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Kích thước output (default: 512)"
    )
    parser.add_argument(
        "--no_enhance",
        action="store_true",
        help="Tắt enhancement (brightness/contrast/sharpness)"
    )
    parser.add_argument(
        "--stretch",
        action="store_true",
        help="Stretch ảnh thay vì giữ tỷ lệ + padding (có thể bị méo)"
    )
    
    args = parser.parse_args()
    
    preprocess_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=args.size,
        enhance=not args.no_enhance,
        keep_aspect_ratio=not args.stretch
    )


if __name__ == "__main__":
    main()
