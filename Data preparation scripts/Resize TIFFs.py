import os
from PIL import Image
from pathlib import Path

# Disable decompression bomb protection for large images
Image.MAX_IMAGE_PIXELS = None

def resize_tiff_to_target(input_tiff, output_tiff, target_width, target_height, is_label_map=False):
    """
    Resize a TIFF image to target dimensions.
    
    Args:
        input_tiff: Path to input TIFF file
        output_tiff: Path to output TIFF file
        target_width: Target width in pixels
        target_height: Target height in pixels
        is_label_map: If True, uses NEAREST resampling (for class labels).
                     If False, uses LANCZOS resampling (for satellite images).
    """
    print(f"\nProcessing: {Path(input_tiff).name}")
    print("-" * 60)
    
    try:
        # Open image
        img = Image.open(input_tiff)
        original_size = img.size
        print(f"Original size: {original_size[0]} x {original_size[1]} pixels")
        print(f"Target size: {target_width} x {target_height} pixels")
        
        # Choose resampling method
        if is_label_map:
            # For label maps: use NEAREST to preserve exact class values
            resampling = Image.NEAREST
            print(f"Mode: Label map (using NEAREST resampling)")
        else:
            # For satellite images: use LANCZOS for best quality
            resampling = Image.LANCZOS
            print(f"Mode: Satellite image (using LANCZOS resampling)")
        
        print(f"Image mode: {img.mode}")
        
        # Resize
        print("Resizing... (this may take a while for large images)")
        resized_img = img.resize((target_width, target_height), resampling)
        
        # Save
        print(f"Saving to: {output_tiff}")
        resized_img.save(output_tiff, compression='lzw')
        
        print(f"✓ Successfully saved: {Path(output_tiff).name}")
        print(f"  Final size: {resized_img.size[0]} x {resized_img.size[1]} pixels")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return False


def resize_tiff_pair(label_tiff, satellite_tiff, output_folder, target_size=None):
    """
    Resize a pair of TIFF images (label map and satellite) to the same size.
    
    Args:
        label_tiff: Path to label/vector map TIFF
        satellite_tiff: Path to satellite image TIFF
        output_folder: Folder where resized images will be saved
        target_size: Tuple (width, height). If None, uses size of satellite image.
    """
    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TIFF PAIR RESIZING")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(label_tiff):
        print(f"ERROR: Label TIFF not found: {label_tiff}")
        return False
    
    if not os.path.exists(satellite_tiff):
        print(f"ERROR: Satellite TIFF not found: {satellite_tiff}")
        return False
    
    # Get target size from satellite image if not specified
    if target_size is None:
        print("\nReading satellite image dimensions...")
        sat_img = Image.open(satellite_tiff)
        target_size = sat_img.size
        sat_img.close()
        print(f"Target size will be: {target_size[0]} x {target_size[1]} pixels (from satellite image)")
    
    target_width, target_height = target_size
    
    # Generate output filenames
    label_output = output_path / f"{Path(label_tiff).stem}_resized.tif"
    satellite_output = output_path / f"{Path(satellite_tiff).stem}_resized.tif"
    
    print(f"\nOutput folder: {output_folder}")
    
    # Resize label map (using NEAREST to preserve class values)
    print("\n" + "=" * 60)
    print("STEP 1/2: Resizing LABEL MAP")
    print("=" * 60)
    success1 = resize_tiff_to_target(
        label_tiff, 
        label_output, 
        target_width, 
        target_height, 
        is_label_map=True
    )
    
    # Resize satellite image (using LANCZOS for quality)
    print("\n" + "=" * 60)
    print("STEP 2/2: Resizing SATELLITE IMAGE")
    print("=" * 60)
    success2 = resize_tiff_to_target(
        satellite_tiff, 
        satellite_output, 
        target_width, 
        target_height, 
        is_label_map=False
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if success1 and success2:
        print("✓ Both images resized successfully!")
        print(f"\nResized files saved in: {output_folder}")
        print(f"  - {label_output.name}")
        print(f"  - {satellite_output.name}")
        print(f"\nBoth images are now: {target_width} x {target_height} pixels")
        return True
    else:
        print("✗ Some errors occurred during resizing")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TIFF PAIR RESIZER FOR SEAMLESS SATELLITE-IMAGE SYNTHESIS")
    print("=" * 60)
    
    # ===== CONFIGURE YOUR PATHS HERE =====
    label_tiff = r"F:\Miguel\Documents\Maestria\Erasmus Mundus\3er Semestre\Thesis UN-GAN\Datasets_Africa\Test2_13-11-25\Pre-resize\map\labels_final\map_final_v2.tif"
    
    satellite_tiff = r"F:\Miguel\Documents\Maestria\Erasmus Mundus\3er Semestre\Thesis UN-GAN\Datasets_Africa\Test2_13-11-25\real\real.tif"
    
    output_folder = r"F:\Miguel\Documents\Maestria\Erasmus Mundus\3er Semestre\Thesis UN-GAN\Datasets_Africa\Test2_13-11-25\map"
    
    # If you want a specific size instead of matching satellite size, uncomment and set:
    # target_size = (32000, 32000)  # (width, height)
    target_size = None  # None = use satellite image size
    # =====================================
    
    # Check if files exist
    if not os.path.exists(label_tiff):
        print(f"\n✗ ERROR: Label TIFF not found")
        print(f"  Path: {label_tiff}")
        print("\nPlease update the 'label_tiff' path in the script.")
        input("\nPress Enter to exit...")
    elif not os.path.exists(satellite_tiff):
        print(f"\n✗ ERROR: Satellite TIFF not found")
        print(f"  Path: {satellite_tiff}")
        print("\nPlease update the 'satellite_tiff' path in the script.")
        input("\nPress Enter to exit...")
    else:
        print(f"\nLabel TIFF: {Path(label_tiff).name}")
        print(f"Satellite TIFF: {Path(satellite_tiff).name}")
        print(f"Output folder: {output_folder}")
        
        user_input = input("\nDo you want to proceed? (yes/no): ").strip().lower()
        
        if user_input in ['yes', 'y']:
            resize_tiff_pair(label_tiff, satellite_tiff, output_folder, target_size)
        else:
            print("Operation cancelled.")
    
    input("\nPress Enter to exit...")