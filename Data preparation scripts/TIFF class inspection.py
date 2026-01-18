import numpy as np
from PIL import Image
import os

# Disable decompression bomb protection for large images
Image.MAX_IMAGE_PIXELS = None

def inspect_tiff_values(tiff_path):
    """
    Inspect and display detailed information about TIFF pixel values.
    
    Args:
        tiff_path: Path to TIFF file
    """
    print("=" * 60)
    print("TIFF VALUE INSPECTOR")
    print("=" * 60)
    
    print(f"\nFile: {tiff_path}")
    print("-" * 60)
    
    try:
        # Open TIFF image
        print("\nLoading TIFF file...")
        img = Image.open(tiff_path)
        
        width, height = img.size
        print(f"\nImage dimensions: {width} x {height} pixels")
        print(f"Image mode: {img.mode}")
        
        # Convert to numpy array
        print("\nConverting to array...")
        img_array = np.array(img)
        
        print(f"Array shape: {img_array.shape}")
        print(f"Array dtype: {img_array.dtype}")
        
        # Get unique values
        print("\nAnalyzing pixel values...")
        unique_values = np.unique(img_array)
        
        print("\n" + "=" * 60)
        print("VALUE STATISTICS")
        print("=" * 60)
        print(f"Minimum value: {img_array.min()}")
        print(f"Maximum value: {img_array.max()}")
        print(f"Number of unique values: {len(unique_values)}")
        
        print("\n" + "-" * 60)
        print("UNIQUE VALUES AND THEIR PIXEL COUNTS:")
        print("-" * 60)
        
        # Count pixels for each unique value
        for value in unique_values:
            count = np.sum(img_array == value)
            percentage = (count / img_array.size) * 100
            print(f"Value {value:3d}: {count:12d} pixels ({percentage:6.2f}%)")
        
        # Check if values look like they need remapping
        print("\n" + "=" * 60)
        print("ANALYSIS")
        print("=" * 60)
        
        if len(unique_values) <= 10:
            print(f"\n✓ This appears to be a label/classification map")
            print(f"  (only {len(unique_values)} unique values)")
            
            if img_array.min() == 0:
                print(f"\n✓ Values already start from 0 (correct for training)")
                print(f"  Value range: 0 to {img_array.max()}")
            elif img_array.min() == 1:
                print(f"\n⚠ Values start from 1 (should start from 0)")
                print(f"  You need to subtract 1 from all values")
                print(f"  Current range: 1 to {img_array.max()}")
                print(f"  Desired range: 0 to {img_array.max() - 1}")
            else:
                print(f"\n⚠ Values don't start from 0 or 1")
                print(f"  Current range: {img_array.min()} to {img_array.max()}")
                print(f"  This might be a different encoding scheme")
        else:
            print(f"\n⚠ Large number of unique values ({len(unique_values)})")
            print(f"  This might be:")
            print(f"  - A continuous raster (not a classification)")
            print(f"  - RGB image stored incorrectly")
            print(f"  - A classification with color values instead of class IDs")
        
        # If it's RGB, check if we need to extract a single band
        if len(img_array.shape) == 3:
            print(f"\n⚠ This is a multi-band image (shape: {img_array.shape})")
            print(f"  Number of bands: {img_array.shape[2]}")
            print(f"\n  Checking if all bands have the same values...")
            
            if img_array.shape[2] >= 3:
                band_0 = img_array[:, :, 0]
                band_1 = img_array[:, :, 1]
                band_2 = img_array[:, :, 2]
                
                if np.array_equal(band_0, band_1) and np.array_equal(band_1, band_2):
                    print(f"  ✓ All bands are identical - can extract single band")
                    print(f"\n  Values in band 0:")
                    unique_band = np.unique(band_0)
                    print(f"    Unique values: {unique_band}")
                    print(f"    Range: {band_0.min()} to {band_0.max()}")
                else:
                    print(f"  ✗ Bands have different values - this is a true RGB image")
        
        print("\n" + "=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TIFF VALUE INSPECTOR")
    print("Check what values your TIFF actually contains")
    print("=" * 60)
    
    # ===== CONFIGURE YOUR PATH HERE =====
    tiff_path = r"F:\Miguel\Documents\Maestria\Erasmus Mundus\3er Semestre\Thesis UN-GAN\Datasets_Africa\Test2_13-11-25\Pre-resize\map\labels_final\map_final_v2.tif"
    
    if not os.path.exists(tiff_path):
        print(f"\n✗ ERROR: File not found")
        print(f"  Path: {tiff_path}")
        print("\nPlease update the 'tiff_path' in the script.")
    else:
        inspect_tiff_values(tiff_path)
    
    input("\nPress Enter to exit...")