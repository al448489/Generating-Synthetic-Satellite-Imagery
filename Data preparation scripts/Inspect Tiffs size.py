import sys
import os
from PIL import Image, UnidentifiedImageError

def get_tiff_dimensions(filepath):
    """
    Tries to open a TIFF file and return its dimensions (width, height).
    Returns (None, None) on failure.
    """
    # 1. Check if the file exists
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return None, None
        
    try:
        # 2. Open the image. 
        # Image.open() is "lazy" - it only reads the header,
        # not the full image data. This is very fast.
        with Image.open(filepath) as img:
            # 3. Return the .size attribute, which is (width, height)
            return img.size 
    except UnidentifiedImageError:
        print(f"Error: Cannot identify '{filepath}' as an image. It might be corrupt or not a valid TIFF.")
        return None, None
    except IOError as e:
        print(f"Error: An I/O error occurred while reading '{filepath}': {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred with '{filepath}': {e}")
        return None, None

def main():
    """
    Main function to check and compare file dimensions.
    """
    # === EDIT THIS SECTION ===
    # Paste the full paths to your two TIFF files here.
    # Use r"..." (raw string) on Windows to avoid issues with backslashes.
    # Example Windows: file1_path = r"C:\Users\YourUser\Desktop\map_v1.tif"
    # Example Mac/Linux: file1_path = "/home/youruser/documents/map_v1.tif"
    
    file1_path = r"F:\Miguel\Documents\Maestria\Erasmus Mundus\3er Semestre\Thesis UN-GAN\Datasets_Africa\Large Training - FINAL\Synth_map\TIFFs\map\map.tif"
    file2_path = r"F:\Miguel\Documents\Maestria\Erasmus Mundus\3er Semestre\Thesis UN-GAN\Datasets_Africa\Large Training - FINAL\Synth_map\TIFFs\ins\ins.tif"
    # =========================

    print("--- TIFF Dimension Inspector ---")
    
    print("--- Inspecting Files ---")
    
    # Get dimensions for both files
    w1, h1 = get_tiff_dimensions(file1_path)
    w2, h2 = get_tiff_dimensions(file2_path)
    
    print("\n--- Results ---")
    
    # Print results for file 1
    if w1 is not None:
        print(f"File 1: {os.path.basename(file1_path)}")
        print(f"  -> Dimensions: {w1} (width) x {h1} (height) pixels")
    
    # Print results for file 2
    if w2 is not None:
        print(f"File 2: {os.path.basename(file2_path)}")
        print(f"  -> Dimensions: {w2} (width) x {h2} (height) pixels")
        
    # Compare the dimensions
    if w1 is not None and w2 is not None:
        if w1 == w2 and h1 == h2:
            print("\n✅ The files have identical dimensions.")
        else:
            print("\n❌ The files have different dimensions.")

# Standard Python entry point
if __name__ == "__main__":
    main()