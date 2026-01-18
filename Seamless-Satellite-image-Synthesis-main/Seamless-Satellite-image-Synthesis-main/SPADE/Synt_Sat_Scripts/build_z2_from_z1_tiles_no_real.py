import argparse
import math
import os
from PIL import Image


def read_pngs_sorted(dir_path):
    fnames = [f for f in os.listdir(dir_path) if f.lower().endswith('.png')]
    fnames.sort()
    return fnames


def ensure_sets(map_dir, ins_dir, synth_dir, pseudo_real_dir=None):
    mp = set(read_pngs_sorted(map_dir))
    ins = set(read_pngs_sorted(ins_dir))
    syn = set(read_pngs_sorted(synth_dir))
    if pseudo_real_dir:
        preal = set(read_pngs_sorted(pseudo_real_dir))
        common = sorted(list(mp & ins & syn & preal))
    else:
        # no real imagery; intersect map/ins/synth only
        common = sorted(list(mp & ins & syn))
    if not common:
        raise SystemExit("No common filenames across required dirs. Check inputs.")
    return common


def parse_xy_from_name(name):
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
        if y < height and x < width:
            grid[y][x] = fname
    ordered = []
    for y in range(height):
        for x in range(width):
            if grid[y][x] is not None:
                ordered.append(grid[y][x])
    return ordered, width, height, grid


def main():
    p = argparse.ArgumentParser(description="Build z2 tiles from z1 tiles WITHOUT real z1 imagery (use synthetic as pseudo-real or blank).")
    p.add_argument('--z1_map_dir', required=True)
    p.add_argument('--z1_ins_dir', required=True)
    p.add_argument('--z1_synth_dir', required=True, help='Directory with z1 synthesized tiles (choose one style if multiple).')
    p.add_argument('--z1_pseudo_real_dir', default=None, help='Optional: directory with pseudo-real tiles (e.g., chosen synthetic style). If omitted, synthetic is reused as real.')
    p.add_argument('--z2_out_root', required=True, help='Output root for z2 with subdirs real/map/ins/guidance')
    p.add_argument('--block', type=int, default=3, help='Grouping factor S: SxS z1 tiles -> 1 z2 tile')
    p.add_argument('--tile_size', type=int, default=256)
    p.add_argument('--guidance_size', type=int, default=64)
    p.add_argument('--tiles_per_row', type=int, default=None, help='If coordinates absent, explicit tiles per row.')
    p.add_argument('--order', choices=['row','col'], default='row', help='Fallback ordering when no coordinates.')
    p.add_argument('--name_mode', choices=['seq','coord'], default='coord')
    p.add_argument('--coord_digits', type=int, default=None)
    p.add_argument('--blank_real', action='store_true', help='If set, create blank real tiles instead of using synthetic as pseudo-real.')
    args = p.parse_args()

    t = args.tile_size
    g = args.guidance_size
    S = args.block
    big = S * t

    os.makedirs(os.path.join(args.z2_out_root, 'real'), exist_ok=True)
    os.makedirs(os.path.join(args.z2_out_root, 'map'), exist_ok=True)
    os.makedirs(os.path.join(args.z2_out_root, 'ins'), exist_ok=True)
    os.makedirs(os.path.join(args.z2_out_root, 'guidance'), exist_ok=True)

    common = ensure_sets(args.z1_map_dir, args.z1_ins_dir, args.z1_synth_dir, args.z1_pseudo_real_dir)
    N = len(common)
    print(f"Found {N} common z1 tiles")

    ordered, Wd, Hd, grid = reorder_with_coordinates(common)
    if ordered is not None:
        common_ordered = ordered
        W = Wd
        H = Hd
        name_grid = grid
        print(f"Coordinate pattern detected: width={W}, height={H}")
    else:
        if args.tiles_per_row is not None:
            W = args.tiles_per_row
            H = math.ceil(N / W)
            print(f"Using provided tiles_per_row={W}; height={H}")
            name_grid = [[None for _ in range(W)] for __ in range(H)]
            k = 0
            for y in range(H):
                for x in range(W):
                    if k < N:
                        name_grid[y][x] = common[k]
                        k += 1
            common_ordered = common
        else:
            approx_w = max(S, int(math.sqrt(N)))
            W = approx_w
            H = math.ceil(N / W)
            print(f"WARNING: No coordinates; approximated width={W}, height={H}. Provide --tiles_per_row for exact layout.")
            # Build order list
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
            name_grid = [[None for _ in range(W)] for __ in range(H)]
            k = 0
            for y in range(H):
                for x in range(W):
                    if k < len(order_list):
                        name_grid[y][x] = order_list[k]
                        k += 1

    # Determine padding for coord naming
    pad_x = pad_y = 0
    if args.name_mode == 'coord':
        sample = None
        for row in name_grid:
            for cell in row:
                if cell:
                    sample = cell
                    break
            if sample:
                break
        if sample:
            base = os.path.splitext(sample)[0]
            parts = base.split('_')
            if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
                pad_x = len(parts[-2])
                pad_y = len(parts[-1])
        if pad_x == 0:
            pad_x = len(str(max(1, W//S)))
        if pad_y == 0:
            pad_y = len(str(max(1, H//S)))
        if args.coord_digits is not None:
            pad_x = pad_y = args.coord_digits

    pseudo_real_dir = args.z1_pseudo_real_dir or args.z1_synth_dir
    use_blank = args.blank_real
    if use_blank:
        print("Blank real tiles will be created (black).")
    else:
        if args.z1_pseudo_real_dir:
            print("Using provided pseudo-real directory for real tiles.")
        else:
            print("Reusing synthetic tiles as pseudo-real for z2 'real'.")

    count = 0
    for by in range(0, H, S):
        for bx in range(0, W, S):
            real_big = Image.new('RGB', (S*t, S*t))
            map_big  = Image.new('L',   (S*t, S*t))
            ins_big  = Image.new('L',   (S*t, S*t))
            syn_big  = Image.new('RGB', (S*t, S*t))
            for j in range(S):
                for i in range(S):
                    gx = bx + i
                    gy = by + j
                    fname = None if gx >= W or gy >= H else name_grid[gy][gx]
                    if fname is None:
                        blank_rgb = Image.new('RGB', (t, t), (0,0,0))
                        blank_l   = Image.new('L', (t, t), 0)
                        real_big.paste(blank_rgb,(i*t,j*t))
                        map_big.paste(blank_l,(i*t,j*t))
                        ins_big.paste(blank_l,(i*t,j*t))
                        syn_big.paste(blank_rgb,(i*t,j*t))
                        continue
                    try:
                        m = Image.open(os.path.join(args.z1_map_dir, fname)).convert('L')
                        ins = Image.open(os.path.join(args.z1_ins_dir, fname)).convert('L')
                        s = Image.open(os.path.join(args.z1_synth_dir, fname)).convert('RGB')
                        if use_blank:
                            r = Image.new('RGB', (t, t), (0,0,0))
                        else:
                            r = Image.open(os.path.join(pseudo_real_dir, fname)).convert('RGB')
                    except Exception as e:
                        print(f"Read error {fname}: {e}; inserting blank")
                        r = Image.new('RGB', (t, t), (0,0,0))
                        m = Image.new('L', (t, t), 0)
                        ins = Image.new('L', (t, t), 0)
                        s = Image.new('RGB', (t, t), (0,0,0))
                    real_big.paste(r,(i*t,j*t))
                    map_big.paste(m,(i*t,j*t))
                    ins_big.paste(ins,(i*t,j*t))
                    syn_big.paste(s,(i*t,j*t))

            # Downsample
            real_out = real_big.resize((t, t), Image.LANCZOS)
            map_out  = map_big.resize((t, t), Image.NEAREST)
            ins_out  = ins_big.resize((t, t), Image.NEAREST)
            guid_out = syn_big.resize((g, g), Image.LANCZOS)

            if args.name_mode == 'coord':
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

    print(f"Wrote z2 tiles: {count}")
    if args.name_mode == 'coord':
        print(f"Coordinate naming padding: X={pad_x}, Y={pad_y}")
    if use_blank:
        print("NOTE: Real tiles are blank; training requiring real texture supervision will not be effective.")
    else:
        print("Pseudo-real tiles populated from synthetic outputs.")


if __name__ == '__main__':
    main()
