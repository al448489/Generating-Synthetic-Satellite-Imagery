import os
from PIL import Image
from pathlib import Path

# Disable decompression bomb protection for large satellite images
Image.MAX_IMAGE_PIXELS = None

def tiff_to_png(tiff_path, output_folder):
    """
    Convert a TIFF image to a single PNG-32 image (no tiling or cropping).
    
    Args:
        tiff_path: Path to the input TIFF file
        output_folder: Folder where the PNG will be saved
    """
    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading TIFF file: {tiff_path}")
    
    try:
        # Open TIFF image
        img = Image.open(tiff_path)
        
        # Convert to RGBA (PNG-32) if not already
        if img.mode != 'RGBA':
            print(f"Converting from {img.mode} to RGBA (PNG-32)...")
            img = img.convert('RGBA')
        
        width, height = img.size
        print(f"Image size: {width}x{height} pixels")
        print(f"Image mode: {img.mode}")
        
        # Get base filename
        base_name = Path(tiff_path).stem
        png_filename = f"{base_name}.png"
        png_path = output_path / png_filename
        
        # Save as PNG-32
        print(f"Saving PNG file to: {png_path}")
        img.save(png_path, 'PNG', compress_level=6)
        
        print(f"\n{'='*60}")
        print(f"Success! PNG image created: {png_path}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("TIFF to PNG-32 Converter (no tiling)")
    print("="*60)
    
    # ===== CONFIGURE YOUR PATHS HERE =====
    tiff_file = r"F:\Miguel\Documents\Maestria\Erasmus Mundus\3er Semestre\Thesis UN-GAN\Datasets_Africa\Test2_13-11-25\map\final\map.tif"
    output_folder = r"F:\Miguel\Documents\Maestria\Erasmus Mundus\3er Semestre\Thesis UN-GAN\Datasets_Africa\Test2_13-11-25\map"
    # =====================================
    
    if not os.path.exists(tiff_file):
        print(f"ERROR: File not found: {tiff_file}")
        input("\nPress Enter to exit...")
    else:
        print(f"Input file: {tiff_file}")
        print(f"Output folder: {output_folder}\n")
        
        user_input = input("Do you want to proceed? (yes/no): ").strip().lower()
        
        if user_input in ['yes', 'y']:
            tiff_to_png(tiff_file, output_folder)
        else:
            print("Operation cancelled.")
    
    input("\nPress Enter to exit...")
