import os
from PIL import Image
import torch
from typing import List, Tuple
from SPADE.data.base_dataset import get_params, get_transform


class LabelInstanceOnlyDataset:
    """Minimal dataset wrapper providing (label, instance) tensors.

    It mirrors transformation logic of the original pix2pix dataset but
    intentionally omits the real satellite image. A dummy RGB tensor can
    be produced when needed (e.g., for VAE or placeholder).
    """

    def __init__(self, label_dir: str, instance_dir: str, opt, device: torch.device):
        self.label_dir = label_dir
        self.instance_dir = instance_dir
        self.opt = opt
        self.device = device
        self.label_paths = self._gather_paths(label_dir)
        self.instance_paths = self._gather_paths(instance_dir)
        if len(self.instance_paths) and len(self.instance_paths) != len(self.label_paths):
            raise ValueError("Instance and label counts differ; ensure matching filenames.")

    def _gather_paths(self, root: str) -> List[str]:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Directory not found: {root}")
        exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        files = [os.path.join(root, f) for f in sorted(os.listdir(root)) if os.path.splitext(f)[1].lower() in exts]
        return files

    def __len__(self):
        return len(self.label_paths)

    def _load_and_transform(self, path: str, nearest: bool = False):
        img = Image.open(path)
        params = get_params(self.opt, img.size)
        transform = get_transform(self.opt, params, method=Image.NEAREST if nearest else Image.BICUBIC, normalize=not nearest)
        tensor = transform(img)
        return tensor, params

    def get_item(self, index: int):
        label_path = self.label_paths[index]
        label_img = Image.open(label_path)
        params = get_params(self.opt, label_img.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label_img) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # unknown class handling

        if len(self.instance_paths):
            instance_path = self.instance_paths[index]
            inst_img = Image.open(instance_path)
            instance_tensor = transform_label(inst_img) * 255
            instance_tensor = instance_tensor.long()
        else:
            instance_tensor = torch.zeros_like(label_tensor).long()

        # Dummy image tensor (unused for pure SPADE forward without VAE)
        dummy_image = torch.zeros(3, label_tensor.shape[1], label_tensor.shape[2])
        return {
            'label': label_tensor.long(),
            'instance': instance_tensor,
            'image': dummy_image,
            'path': label_path
        }
