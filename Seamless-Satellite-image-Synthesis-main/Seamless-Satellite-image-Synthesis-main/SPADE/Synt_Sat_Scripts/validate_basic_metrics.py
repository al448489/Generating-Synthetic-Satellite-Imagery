import os
import argparse
from PIL import Image
import numpy as np
import json

def parse_args():
    p = argparse.ArgumentParser("Basic validation metrics for generated satellite imagery")
    p.add_argument('--generated_dir', type=str, required=True)
    p.add_argument('--label_dir', type=str, required=True, help='Semantic label maps used for generation')
    p.add_argument('--output_json', type=str, default='validation_metrics.json')
    return p.parse_args()

def load_images(folder, exts={'.png', '.jpg', '.jpeg'}):
    imgs = []
    paths = []
    for f in sorted(os.listdir(folder)):
        if os.path.splitext(f)[1].lower() in exts:
            path = os.path.join(folder, f)
            try:
                imgs.append(Image.open(path))
                paths.append(path)
            except Exception:
                pass
    return paths, imgs

def color_stats(imgs):
    stats = {'mean': [], 'std': [], 'hist': []}
    for im in imgs:
        arr = np.array(im.convert('RGB'), dtype=np.float32) / 255.0
        stats['mean'].append(arr.mean(axis=(0,1)).tolist())
        stats['std'].append(arr.std(axis=(0,1)).tolist())
        # Simple histogram (16 bins per channel)
        hists = []
        for c in range(3):
            h, _ = np.histogram(arr[..., c], bins=16, range=(0,1), density=True)
            hists.append(h.tolist())
        stats['hist'].append(hists)
    # Aggregate
    agg = {
        'mean_mean': np.mean(stats['mean'], axis=0).tolist(),
        'mean_std': np.mean(stats['std'], axis=0).tolist(),
    }
    return stats, agg

def label_distribution(label_imgs):
    dist = None
    for im in label_imgs:
        arr = np.array(im, dtype=np.int32)
        flat = arr.flatten()
        if dist is None:
            max_label = flat.max()
            dist = np.zeros(max_label+1, dtype=np.int64)
        # Extend if necessary
        if flat.max() >= dist.shape[0]:
            new = np.zeros(flat.max()+1, dtype=np.int64)
            new[:dist.shape[0]] = dist
            dist = new
        for v in flat:
            dist[v] += 1
    total = dist.sum()
    probs = (dist / total).tolist()
    return {'counts': dist.tolist(), 'probs': probs}

def main():
    a = parse_args()
    gen_paths, gen_imgs = load_images(a.generated_dir)
    lbl_paths, lbl_imgs = load_images(a.label_dir)
    color_detail, color_agg = color_stats(gen_imgs)
    label_dist = label_distribution(lbl_imgs)
    report = {
        'num_generated': len(gen_imgs),
        'color_detail': color_detail,
        'color_aggregate': color_agg,
        'label_distribution': label_dist
    }
    with open(a.output_json, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved metrics to {a.output_json}")

if __name__ == '__main__':
    main()
