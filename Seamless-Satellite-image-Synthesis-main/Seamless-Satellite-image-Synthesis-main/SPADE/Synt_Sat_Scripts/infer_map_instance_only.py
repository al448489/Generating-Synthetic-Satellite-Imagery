import os
import sys
import argparse
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Flexible import handling so script runs from project root OR inside SPADE.
# Adds project root (parent of SPADE) to sys.path; then attempts canonical
# package imports. Falls back to relative imports if needed.
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Correct: parent of script dir is the SPADE folder
SPADE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))  # .../SPADE
PROJECT_ROOT = os.path.abspath(os.path.join(SPADE_DIR, '..'))  # repo root
for p in (PROJECT_ROOT, SPADE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from SPADE.options.base_options import BaseOptions
    import SPADE.models as models
    from SPADE.models.pix2pix_model import Pix2PixModel
    from SPADE.Synt_Sat_Scripts.label_instance_dataset import LabelInstanceOnlyDataset
except ModuleNotFoundError:
    # Fallback attempt: add SPADE_DIR explicitly again and retry relative imports
    if SPADE_DIR not in sys.path:
        sys.path.insert(0, SPADE_DIR)
    try:
        from options.base_options import BaseOptions  # type: ignore
        import models as models  # type: ignore
        from models.pix2pix_model import Pix2PixModel  # type: ignore
        from Synt_Sat_Scripts.label_instance_dataset import LabelInstanceOnlyDataset  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError("Failed to import SPADE modules. Run script from repo root (the directory containing 'SPADE') or ensure 'SPADE/__init__.py' exists.") from e


def build_semantics(label_tensor: torch.Tensor, instance_tensor: torch.Tensor, opt, device):
    """Replicates preprocess_input semantic + edge concatenation.

    Accepts label/instance as [B,1,H,W] or [B,H,W] (or [1,H,W]); converts accordingly.
    """
    def ensure_b1hw(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 4:
            # assume [B,C,H,W]; if C>1, take first channel for labels/instances
            if t.shape[1] != 1:
                t = t[:, :1, :, :]
            return t
        elif t.ndim == 3:
            # Could be [C,H,W] or [B,H,W]; standardize to [B,1,H,W]
            if t.shape[0] != 1 or t.shape[1] != t.shape[1]:
                # Assume [C,H,W] -> take first channel
                t = t[:1, :, :]
            return t.unsqueeze(0)
        elif t.ndim == 2:
            return t.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected tensor shape for semantic building: {t.shape}")

    label_idx = ensure_b1hw(label_tensor.long().to(device))  # [B,1,H,W]
    bs, _, h, w = label_idx.shape
    nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0)
    one_hot = torch.zeros(bs, nc, h, w, device=device)
    one_hot.scatter_(1, label_idx, 1.0)

    if not opt.no_instance:
        inst_map = ensure_b1hw(instance_tensor.to(device))  # [B,1,H,W]
        inst_map = inst_map[:, 0, :, :]  # [B,H,W]
        edge = torch.zeros_like(inst_map, dtype=torch.float32)
        edge[:, 1:, :] = edge[:, 1:, :] + (inst_map[:, 1:, :] != inst_map[:, :-1, :]).float()
        edge[:, :-1, :] = edge[:, :-1, :] + (inst_map[:, 1:, :] != inst_map[:, :-1, :]).float()
        edge[:, :, 1:] = edge[:, :, 1:] + (inst_map[:, :, 1:] != inst_map[:, :, :-1]).float()
        edge[:, :, :-1] = edge[:, :, :-1] + (inst_map[:, :, 1:] != inst_map[:, :, :-1]).float()
        edge = edge.clamp(0, 1).unsqueeze(1)  # [B,1,H,W]
        semantics = torch.cat((one_hot, edge), dim=1)
    else:
        semantics = one_hot
    return semantics


def save_image(tensor: torch.Tensor, path: str):
    # tensor expected in [-1,1]
    img = (tensor.detach().cpu().clamp(-1, 1) + 1) / 2.0  # [0,1]
    img = (img * 255).byte()
    arr = img.permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)


def load_opt(checkpoints_dir: str, experiment_name: str):
    pkl_path = os.path.join(checkpoints_dir, experiment_name, 'opt.pkl')
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Could not find options pickle: {pkl_path}")
    import pickle
    opt = pickle.load(open(pkl_path, 'rb'))
    opt.isTrain = False
    return opt


def parse_args():
    parser = argparse.ArgumentParser("Infer synthetic satellite imagery from map + instance only")
    parser.add_argument('--checkpoints_dir', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--which_epoch', type=str, default='latest')
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--instance_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--random_latent', action='store_true', help='Use random latent vectors instead of encoder')
    parser.add_argument('--num_styles', type=int, default=1, help='Number of random style samples per semantic input')
    parser.add_argument('--seed', type=int, default=42)
    # New options for style-specific subfolders & coordinate naming
    parser.add_argument('--style_subfolders', action='store_true', help='Save each style into its own subfolder Style_<k>')
    parser.add_argument('--coord_mode', type=str, default='last2', choices=['last2','first2'], help='How to select coordinate tokens from filename')
    parser.add_argument('--coord_pad', type=int, default=5, help='Zero pad width for coordinate numbers')
    parser.add_argument('--no_coord_parse', action='store_true', help='If set, do not parse coordinates; keep original base name')
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')

    opt = load_opt(args.checkpoints_dir, args.experiment_name)
    opt.checkpoints_dir = args.checkpoints_dir
    opt.which_epoch = args.which_epoch
    opt.gpu_ids = [args.gpu] if args.gpu >= 0 and torch.cuda.is_available() else []
    opt.isTrain = False
    # Ensure instance usage (adjust if you want to ignore instances)
    opt.no_instance = False
    # semantic_nc is normally set in BaseOptions.parse(); we replicate here
    if not hasattr(opt, 'contain_dontcare_label'):
        # Older checkpoints may use contain_dontcare_label or contain_dontcare_label
        setattr(opt, 'contain_dontcare_label', False)
    opt.semantic_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

    # Create model (loads generator weights)
    model: Pix2PixModel = models.create_model(opt)
    model.eval()
    model.to(device)

    # Dataset
    ds = LabelInstanceOnlyDataset(args.label_dir, args.instance_dir, opt, device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Precompute latent size
    z_dim = getattr(opt, 'z_dim', 256)

    for idx in tqdm(range(len(ds)), desc='Generating'):
        sample = ds.get_item(idx)
        # Reshape to batch size 1
        label = sample['label'].unsqueeze(0)  # (1,H,W)
        instance = sample['instance'].unsqueeze(0)  # (1,H,W)
        semantics = build_semantics(label, instance, opt, device)

        generated_list = []
        if args.random_latent:
            for s in range(args.num_styles):
                z = torch.randn(1, z_dim, device=device)
                fake = model.netG(semantics, z=z)
                generated_list.append(fake[0])
        else:
            # Encode z from dummy zeros (may yield low-quality style if VAE used). Provide at least one output.
            dummy_image = torch.zeros(1, 3, semantics.shape[2], semantics.shape[3], device=device)
            data_dict = {'label': label, 'instance': instance, 'image': dummy_image}
            fake = model.forward(data_dict, mode='inference')
            generated_list.append(fake[0])

        base_name = os.path.splitext(os.path.basename(sample['path']))[0]
        # Coordinate parsing helper
        def extract_coords(name: str) -> str:
            if args.no_coord_parse:
                return name
            import re
            nums = re.findall(r'\d+', name)
            if len(nums) < 2:
                return name  # fallback
            if args.coord_mode == 'last2':
                sel = nums[-2:]
            else:
                sel = nums[:2]
            sel = [n.zfill(args.coord_pad) for n in sel]
            return f"{sel[0]}_{sel[1]}"

        coord_name = extract_coords(base_name)

        for k, img_tensor in enumerate(generated_list):
            if args.style_subfolders:
                style_dir = os.path.join(args.output_dir, f"Style_{k+1}")
                os.makedirs(style_dir, exist_ok=True)
                out_path = os.path.join(style_dir, f"{coord_name}.png")
            else:
                # Keep previous flat structure but remove style suffix from filename? Only if coord parsing requested.
                fname = f"{coord_name}_style{k}.png" if not args.style_subfolders else f"{coord_name}.png"
                out_path = os.path.join(args.output_dir, fname)
            save_image(img_tensor, out_path)

    print(f"Done. Images saved to {args.output_dir}")


if __name__ == '__main__':
    main()
