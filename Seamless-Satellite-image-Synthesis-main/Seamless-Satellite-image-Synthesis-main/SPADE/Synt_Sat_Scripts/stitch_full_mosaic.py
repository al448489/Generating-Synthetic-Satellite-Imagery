import argparse
import math
import os
from PIL import Image


def read_pngs_sorted(dir_path):
    fnames = [f for f in os.listdir(dir_path) if f.lower().endswith(".png")]
    fnames.sort()
    return fnames


def parse_xy_from_name(name):
    """Parse coordinates from filenames like '..._X_Y.png'. Returns (x, y) or None."""
    base = os.path.splitext(name)[0]
    parts = base.split("_")
    if len(parts) < 2:
        return None
    x_part = parts[-2]
    y_part = parts[-1]
    if x_part.isdigit() and y_part.isdigit():
        return int(x_part), int(y_part)
    return None


def reorder_with_coordinates(fnames):
    """Infer grid layout from coordinates embedded in filenames.

    Returns (ordered_list, width, height, grid) or (None, None, None, None)
    if no valid coordinates are found.
    """
    coords = []
    for f in fnames:
        xy = parse_xy_from_name(f)
        if xy is not None:
            coords.append((xy[0], xy[1], f))
    if not coords:
        return None, None, None, None

    max_x = max(c[0] for c in coords)
    max_y = max(c[1] for c in coords)
    width = max_x + 1
    height = max_y + 1

    grid = [[None for _ in range(width)] for __ in range(height)]
    for x, y, fname in coords:
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = fname

    ordered = []
    for y in range(height):
        for x in range(width):
            if grid[y][x] is not None:
                ordered.append(grid[y][x])

    return ordered, width, height, grid


def build_grid_without_coords(fnames, tiles_per_row=None, order="row"):
    """Fallback grid builder when filenames lack coordinates.

    - If tiles_per_row is given, uses that as the width.
    - Otherwise approximates a square layout.
    - order="row" means row-major, "col" means column-major.
    """
    N = len(fnames)
    if N == 0:
        raise SystemExit("No PNG files found in tiles_dir")

    if tiles_per_row is not None:
        W = tiles_per_row
        H = math.ceil(N / W)
    else:
        approx_w = int(math.sqrt(N))
        approx_w = max(1, approx_w)
        W = approx_w
        H = math.ceil(N / W)

    idxs = list(range(N))
    padded = idxs + [N - 1] * (W * H - N)

    order_list = []
    if order == "row":
        k = 0
        for y in range(H):
            for x in range(W):
                order_list.append(fnames[padded[k]])
                k += 1
    else:
        k = 0
        for x in range(W):
            for y in range(H):
                order_list.append(fnames[padded[k]])
                k += 1

    grid = [[None for _ in range(W)] for __ in range(H)]
    k = 0
    for y in range(H):
        for x in range(W):
            if k < len(order_list):
                grid[y][x] = order_list[k]
                k += 1

    return order_list, W, H, grid


def stitch_tiles(tiles_dir, out_path, tiles_per_row=None, order="row"):
    fnames = read_pngs_sorted(tiles_dir)
    if not fnames:
        raise SystemExit(f"No PNG files found in {tiles_dir}")

    # Try to infer grid from coordinates embedded in filenames
    ordered, W, H, grid = reorder_with_coordinates(fnames)
    if ordered is not None:
        print(f"Detected coordinate pattern: width={W}, height={H}")
    else:
        print("No coordinates detected in filenames; falling back to sequential layout.")
        ordered, W, H, grid = build_grid_without_coords(fnames, tiles_per_row, order)
        print(f"Layout: width={W}, height={H}")

    # Determine tile size and mode from the first tile
    first_path = os.path.join(tiles_dir, ordered[0])
    with Image.open(first_path) as im0:
        tile_w, tile_h = im0.size
        mode = im0.mode

    full_w = W * tile_w
    full_h = H * tile_h
    print(f"Creating mosaic of size {full_w} x {full_h} (W x H in pixels)")

    mosaic = Image.new(mode, (full_w, full_h))

    for y in range(H):
        for x in range(W):
            fname = grid[y][x]
            if fname is None:
                continue
            tile_path = os.path.join(tiles_dir, fname)
            try:
                with Image.open(tile_path) as tile:
                    if tile.size != (tile_w, tile_h):
                        tile = tile.resize((tile_w, tile_h), Image.BILINEAR)
                    mosaic.paste(tile, (x * tile_w, y * tile_h))
            except Exception as e:
                print(f"Warning: could not read {tile_path}: {e}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mosaic.save(out_path)
    print(f"Saved full mosaic to {out_path}")



def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stitch 256x256 (or uniform-sized) tiles into a full-resolution mosaic. "
            "If filenames contain '..._X_Y.png', coordinates are used; otherwise, "
            "a sequential grid layout is assumed."
        )
    )
    parser.add_argument("--tiles_dir", required=True,
                        help="Directory containing tile PNGs (e.g., synthesized z2 tiles)")
    parser.add_argument("--out_path", required=True,
                        help="Output path for the stitched full-resolution image (PNG)")
    parser.add_argument("--tiles_per_row", type=int, default=None,
                        help="Optional: tiles per row if filenames lack coordinates")
    parser.add_argument("--order", choices=["row", "col"], default="row",
                        help="Sequential ordering when coordinates are absent (row or column major)")

    args = parser.parse_args()
    stitch_tiles(args.tiles_dir, args.out_path, args.tiles_per_row, args.order)


if __name__ == "__main__":
    main()
