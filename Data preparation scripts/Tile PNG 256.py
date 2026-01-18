from PIL import Image
import os

# === USER CONFIGURATION ===
# 👇 Put the full path to your PNG file or folder here
INPUT_PATH = r"F:\Miguel\Documents\Maestria\Erasmus Mundus\3er Semestre\Thesis UN-GAN\Datasets_Africa\Test2_13-11-25\full PNGs\map\map.png" # or a folder like r"C:\path\to\images"
OUTPUT_DIR = r"F:\Miguel\Documents\Maestria\Erasmus Mundus\3er Semestre\Thesis UN-GAN\Datasets_Africa\Test2_13-11-25\Tiles\map"  # Where tiles will be saved
TILE_SIZE = 256        # Tile width/height in pixels

def tile_image(image_path, tile_size=TILE_SIZE, output_dir=OUTPUT_DIR):
    img = Image.open(image_path).convert("RGBA")
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    width, height = img.size

    os.makedirs(output_dir, exist_ok=True)

    x_tiles = (width + tile_size - 1) // tile_size
    y_tiles = (height + tile_size - 1) // tile_size

    print(f"Tiling {img_name}: {x_tiles}×{y_tiles} tiles...")

    for y in range(y_tiles):
        for x in range(x_tiles):
            left = x * tile_size
            upper = y * tile_size
            right = min(left + tile_size, width)
            lower = min(upper + tile_size, height)

            tile = img.crop((left, upper, right, lower))
            tile_filename = f"{img_name}_{x}_{y}.png"
            tile.save(os.path.join(output_dir, tile_filename), "PNG")

    print(f"✅ Saved tiles for {img_name} in '{output_dir}'\n")

def process_input_path(input_path):
    if os.path.isdir(input_path):
        for file in os.listdir(input_path):
            if file.lower().endswith(".png"):
                tile_image(os.path.join(input_path, file))
    elif os.path.isfile(input_path):
        tile_image(input_path)
    else:
        print(f"⚠️ Invalid path: {input_path}")

if __name__ == "__main__":
    process_input_path(INPUT_PATH)
