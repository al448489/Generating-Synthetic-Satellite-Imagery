import argparse
import math
import os
import re
from PIL import Image


def read_pngs_sorted(dir_path):
    fnames = [f for f in os.listdir(dir_path) if f.lower().endswith('.png')]
    fnames.sort()
    return fnames


def ensure_pairs(real_dir, map_dir, ins_dir, synth_dir):
    real = set(read_pngs_sorted(real_dir))
    mp = set(read_pngs_sorted(map_dir))
    ins = set(read_pngs_sorted(ins_dir))
    syn = set(read_pngs_sorted(synth_dir))
    common = sorted(list(real & mp & ins & syn))
    if not common:
        raise SystemExit("No common filenames across real/map/ins/synth dirs. Check inputs.")
    return common


def parse_xy_from_name(name):
    """Attempt to parse x,y coordinates from a filename following pattern ..._<x>_<y>.png.
    Returns (x,y) or None if pattern not matched."""
    base = os.path.splitext(name)[0]
    parts = base.split('_')
    if len(parts) < 2:
        return None
    x_part = parts[-2]
    y_part = parts[-1]
    if x_part.isdigit() and y_part.isdigit():
        return int(x_part), int(y_part)
    return None


def reorder_with_coordinates(fnames):
    """If coordinate pattern is found, build a 2D grid (height x width) indexed by [y][x].
    Each cell holds a filename or None if missing.
    Returns (ordered_list, width, height, grid). If not found, returns (None, None, None, None)."""
    coords = []
    for f in fnames:
        xy = parse_xy_from_name(f)
        if xy is not None:
            coords.append((xy[0], xy[1], f))
    if not coords:
        return None, None, None, None
    # Determine grid size
    max_x = max(c[0] for c in coords)
    max_y = max(c[1] for c in coords)
    width = max_x + 1
    height = max_y + 1
    # Build mapping
    grid = [[None for _ in range(width)] for __ in range(height)]
    for x, y, fname in coords:
        if y < height and x < width:
            grid[y][x] = fname
    # Some tiles might be missing; fall back by ignoring None entries later
    ordered = []
    for y in range(height):
        for x in range(width):
            if grid[y][x] is not None:
                ordered.append(grid[y][x])
    return ordered, width, height, grid


def main():
    p = argparse.ArgumentParser(description="Build z2 tiles from z1 tiles via mosaic grouping and generate 64x64 guidance from z1 synthesized outputs.")
    p.add_argument('--z1_real_dir', required=True)
    p.add_argument('--z1_map_dir', required=True)
    p.add_argument('--z1_ins_dir', required=True)
    p.add_argument('--z1_synth_dir', required=True, help='Directory with z1 synthesized tiles (e.g., results/.../images/synthesized_image)')
    p.add_argument('--z2_out_root', required=True, help='Output root for z2 with subdirs real/map/ins/guidance')
    p.add_argument('--block', type=int, default=3, help='Grouping factor S (e.g., 3 means 3x3 z1 tiles -> 1 z2 tile)')
    p.add_argument('--tile_size', type=int, default=256)
    p.add_argument('--guidance_size', type=int, default=64)
    p.add_argument('--order', choices=['row', 'col'], default='row', help="Sequential fallback ordering if coordinates not embedded: 'row' = row-major, 'col' = column-major")
    p.add_argument('--tiles_per_row', type=int, default=None, help='Explicit number of tiles per original row (overrides heuristic)')
    p.add_argument('--name_mode', choices=['seq','coord'], default='coord', help="Output naming: 'seq' = 00001.png, 'coord' = X_Y.png based on grouped coordinates")
    p.add_argument('--coord_digits', type=int, default=None, help='Zero padding digits for X and Y in coord mode (auto if omitted)')
    args = p.parse_args()

    t = args.tile_size
    g = args.guidance_size
    S = args.block
    big = S * t

    os.makedirs(os.path.join(args.z2_out_root, 'real'), exist_ok=True)
    os.makedirs(os.path.join(args.z2_out_root, 'map'), exist_ok=True)
    os.makedirs(os.path.join(args.z2_out_root, 'ins'), exist_ok=True)
    os.makedirs(os.path.join(args.z2_out_root, 'guidance'), exist_ok=True)

    common = ensure_pairs(args.z1_real_dir, args.z1_map_dir, args.z1_ins_dir, args.z1_synth_dir)
    N = len(common)
    print(f"Found {N} common z1 tiles")

    # Attempt coordinate-based ordering
    ordered, width_detected, height_detected, coord_grid = reorder_with_coordinates(common)
    if ordered is not None:
        common_ordered = ordered
        W = width_detected
        H = height_detected
        name_grid = coord_grid  # preserve holes as None
        print(f"Detected coordinate pattern: width={W}, height={H}")
    else:
        # Fallback: use tiles_per_row if provided
        if args.tiles_per_row is not None:
            W = args.tiles_per_row
            H = math.ceil(N / W)
            print(f"Using provided tiles_per_row={W}, computed height={H}")
            common_ordered = common
            # Build sequential grid row-major (no holes known)
            name_grid = [[None for _ in range(W)] for __ in range(H)]
            k = 0
            for y in range(H):
                for x in range(W):
                    if k < N:
                        name_grid[y][x] = common[k]
                        k += 1
        else:
            # Last resort: sequential ordering with approximate width via sqrt (may break adjacency across rows)
            approx_w = max(S, int(math.sqrt(N)))
            W = approx_w
            H = math.ceil(N / W)
            print(f"WARNING: Could not detect coordinates. Approximated width={W}, height={H}. Provide --tiles_per_row for exact layout.")
            # Build sequential list respecting chosen order
            idxs = list(range(N))
            padded = idxs + [N-1] * (W * H - N)
            order_list = []
            if args.order == 'row':
                k = 0
                for y in range(H):
                    for x in range(W):
                        order_list.append(common[padded[k]])
                        k += 1
            else:
                k = 0
                for x in range(W):
                    for y in range(H):
                        order_list.append(common[padded[k]])
                        k += 1
            common_ordered = order_list
            # Build grid using order_list
            name_grid = [[None for _ in range(W)] for __ in range(H)]
            k = 0
            for y in range(H):
                for x in range(W):
                    if k < len(order_list):
                        name_grid[y][x] = order_list[k]
                        k += 1

    print(f"Total tiles after ordering reference: {len(common_ordered)}")

    # Do NOT pad holes; leave None where missing so we can insert blanks

    # Determine digit padding for coord mode
    sample_fname = None
    if args.name_mode == 'coord':
        # find first non-None filename
        for row in name_grid:
            for cell in row:
                if cell is not None:
                    sample_fname = cell
                    break
            if sample_fname:
                break
        pad_x = pad_y = 0
        if sample_fname:
            base = os.path.splitext(sample_fname)[0]
            parts = base.split('_')
            if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
                pad_x = len(parts[-2])
                pad_y = len(parts[-1])
        # Fallback to computed grid sizes if parsing failed
        if pad_x == 0:
            pad_x = len(str(W//S))
        if pad_y == 0:
            pad_y = len(str(H//S))
        if args.coord_digits is not None:
            pad_x = pad_y = args.coord_digits

    count = 0
    mosaic_x_max = (W + S - 1) // S
    mosaic_y_max = (H + S - 1) // S
    for by in range(0, H, S):
        for bx in range(0, W, S):
            # Compose SxS mosaics for real/map/ins and synth
            real_big = Image.new('RGB', (big, big))
            map_big  = Image.new('L',   (big, big))  # labels are often single-channel; convert back to 'L'
            ins_big  = Image.new('L',   (big, big))
            syn_big  = Image.new('RGB', (big, big))

            for j in range(S):
                for i in range(S):
                    gx = bx + i
                    gy = by + j
                    # Determine filename; None means hole
                    if gx >= W or gy >= H:
                        fname = None
                    else:
                        fname = name_grid[gy][gx]
                    if fname is None:
                        # Insert blank tiles (black RGB / zero grayscale)
                        real_blank = Image.new('RGB', (t, t), (0, 0, 0))
                        map_blank = Image.new('L', (t, t), 0)
                        ins_blank = Image.new('L', (t, t), 0)
                        synth_blank = Image.new('RGB', (t, t), (0, 0, 0))
                        real_big.paste(real_blank, (i*t, j*t))
                        map_big.paste(map_blank, (i*t, j*t))
                        ins_big.paste(ins_blank, (i*t, j*t))
                        syn_big.paste(synth_blank, (i*t, j*t))
                        continue
                    try:
                        r = Image.open(os.path.join(args.z1_real_dir, fname)).convert('RGB')
                        m = Image.open(os.path.join(args.z1_map_dir,  fname)).convert('L')
                        ins = Image.open(os.path.join(args.z1_ins_dir,  fname)).convert('L')
                        s = Image.open(os.path.join(args.z1_synth_dir, fname)).convert('RGB')
                    except Exception as e:
                        print(f"Read error {fname} in block ({bx},{by}): {e}; inserting blank")
                        r = Image.new('RGB', (t, t), (0, 0, 0))
                        m = Image.new('L', (t, t), 0)
                        ins = Image.new('L', (t, t), 0)
                        s = Image.new('RGB', (t, t), (0, 0, 0))
                    real_big.paste(r,  (i*t, j*t))
                    map_big.paste(m,   (i*t, j*t))
                    ins_big.paste(ins, (i*t, j*t))
                    syn_big.paste(s,   (i*t, j*t))

            # Downsample mosaics for z2 real/map/ins to 256x256
            real_out = real_big.resize((t, t), Image.LANCZOS)
            map_out  = map_big.resize((t, t), Image.NEAREST)   # keep labels crisp
            ins_out  = ins_big.resize((t, t), Image.NEAREST)
            # Guidance: coarse color; downsample synth mosaic to g x g
            guid_out = syn_big.resize((g, g), Image.LANCZOS)

            if args.name_mode == 'coord':
                # block coordinate at lower scale
                mx = bx // S
                my = by // S
                stem = f"{mx:0{pad_x}d}_{my:0{pad_y}d}.png"
            else:
                stem = f"{count:05d}.png"
            real_out.save(os.path.join(args.z2_out_root, 'real', stem))
            map_out.save(os.path.join(args.z2_out_root, 'map', stem))
            ins_out.save(os.path.join(args.z2_out_root, 'ins', stem))
            guid_out.save(os.path.join(args.z2_out_root, 'guidance', stem))
            count += 1

    print(f"Wrote z2 tiles: {count} (real/map/ins/guidance)")
    if args.name_mode == 'coord':
        print(f"Coordinate naming used with padding X={pad_x}, Y={pad_y}. Mosaic grid: {mosaic_x_max} x {mosaic_y_max}")


if __name__ == '__main__':
    main()
