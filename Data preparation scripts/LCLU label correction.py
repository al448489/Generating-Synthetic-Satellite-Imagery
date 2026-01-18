import numpy as np
from PIL import Image
from pathlib import Path
import os

# Disable decompression bomb protection for large images
Image.MAX_IMAGE_PIXELS = None

def clean_and_subtract_one(input_tiff, output_tiff, remove_value=None, replace_with=None):
    """
    Clean unwanted values and subtract 1 from all pixel values in a label TIFF.
    
    Args:
        input_tiff: Path to input TIFF file with values 1-8 (and possibly unwanted values)
        output_tiff: Path to output TIFF file with values 0-7
        remove_value: Value to remove/replace (e.g., 15)
        replace_with: What to replace remove_value with. If None, uses mode (most common value)
    """
    print("=" * 60)
    print("LABEL VALUE CLEANER AND CONVERTER")
    print("=" * 60)
    
    print(f"\nInput file: {input_tiff}")
    print(f"Output file: {output_tiff}")
    print("-" * 60)
    
    try:
        # Open TIFF image
        print("\nLoading TIFF file...")
        img = Image.open(input_tiff)
        
        width, height = img.size
        print(f"Image size: {width} x {height} pixels")
        print(f"Image mode: {img.mode}")
        
        # Convert to numpy array for manipulation
        print("\nConverting to array...")
        img_array = np.array(img)
        
        # Show original value statistics
        unique_values, counts = np.unique(img_array, return_counts=True)
        print(f"\nOriginal unique values and counts:")
        print("-" * 60)
        for val, count in zip(unique_values, counts):
            percentage = (count / img_array.size) * 100
            print(f"  Value {val:3d}: {count:12d} pixels ({percentage:6.2f}%)")
        print("-" * 60)
        
        print(f"\nOriginal min value: {img_array.min()}")
        print(f"Original max value: {img_array.max()}")
        
        # Handle unwanted values
        if remove_value is not None:
            if remove_value in unique_values:
                count_to_remove = np.sum(img_array == remove_value)
                print(f"\n⚠ Found {count_to_remove} pixels with value {remove_value}")
                
                if replace_with is None:
                    # Use the most common valid value (1-8)
                    valid_mask = (img_array >= 1) & (img_array <= 8)
                    valid_values, valid_counts = np.unique(img_array[valid_mask], return_counts=True)
                    replace_with = valid_values[np.argmax(valid_counts)]
                    print(f"  Will replace with most common value: {replace_with}")
                else:
                    print(f"  Will replace with specified value: {replace_with}")
                
                # Replace the unwanted value
                img_array[img_array == remove_value] = replace_with
                print(f"✓ Replaced {count_to_remove} pixels")
        
        # Subtract 1 from all values
        print(f"\nSubtracting 1 from all pixel values...")
        img_array = img_array - 1
        
        # Show new value statistics
        unique_values_new, counts_new = np.unique(img_array, return_counts=True)
        print(f"\nNew unique values and counts:")
        print("-" * 60)
        for val, count in zip(unique_values_new, counts_new):
            percentage = (count / img_array.size) * 100
            print(f"  Value {val:3d}: {count:12d} pixels ({percentage:6.2f}%)")
        print("-" * 60)
        
        print(f"\nNew min value: {img_array.min()}")
        print(f"New max value: {img_array.max()}")
        
        # Verify we have the expected range
        if img_array.min() != 0:
            print(f"\n⚠ WARNING: Minimum value is {img_array.min()}, expected 0")
        
        expected_max = len(unique_values_new) - 1
        if img_array.max() != expected_max:
            print(f"\n⚠ WARNING: Maximum value is {img_array.max()}, expected {expected_max}")
        
        # Convert back to image
        print("\nConverting back to image...")
        img_array = img_array.astype(np.uint8)
        new_img = Image.fromarray(img_array, mode='L')  # Grayscale
        
        # Save the result
        print(f"\nSaving to: {output_tiff}")
        new_img.save(output_tiff, compression='lzw')
        
        print("\n" + "=" * 60)
        print("✓ SUCCESS!")
        print("=" * 60)
        print(f"Final classes: {len(unique_values_new)} (values {img_array.min()} to {img_array.max()})")
        print(f"Saved to: {output_tiff}")
        print(f"\n👉 Use --label_nc {len(unique_values_new)} when training the model")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TIFF LABEL VALUE CLEANER AND CONVERTER")
    print("Removes unwanted values and converts 1-8 to 0-7")
    print("=" * 60)
    
    # ===== CONFIGURE YOUR PATHS HERE =====
    input_tiff = r"F:\Miguel\Documents\Maestria\Erasmus Mundus\3er Semestre\Thesis UN-GAN\Datasets_Africa\Test_Training1\Full Images PNG\PNGs same size\attempt 3\Eth_LCLU_3_resized.tif"
    
    output_tiff = r"F:\Miguel\Documents\Maestria\Erasmus Mundus\3er Semestre\Thesis UN-GAN\Datasets_Africa\Test_Training1\Full Images PNG\PNGs same size\attempt 3\corrected\Eth_Lab_final.tif"
    
    # Value to remove (set to None if not needed)
    remove_value = 15  # Remove value 15
    
    # What to replace it with (None = use most common value)
    replace_with = None  # Will use the most common class (probably class 4)
    # Or specify manually, e.g.: replace_with = 4
    # =====================================
    
    # Check if input file exists
    if not os.path.exists(input_tiff):
        print(f"\n✗ ERROR: Input TIFF not found")
        print(f"  Path: {input_tiff}")
        print("\nPlease update the 'input_tiff' path in the script.")
        input("\nPress Enter to exit...")
    else:
        print(f"\nThis will:")
        print(f"  1. Read: {Path(input_tiff).name}")
        if remove_value is not None:
            print(f"  2. Remove/replace pixels with value {remove_value}")
        print(f"  3. Subtract 1 from all pixel values")
        print(f"  4. Save as: {Path(output_tiff).name}")
        
        user_input = input("\nDo you want to proceed? (yes/no): ").strip().lower()
        
        if user_input in ['yes', 'y']:
            clean_and_subtract_one(input_tiff, output_tiff, remove_value, replace_with)
        else:
            print("Operation cancelled.")
    
    input("\nPress Enter to exit...")