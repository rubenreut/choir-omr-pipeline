#!/usr/bin/env python3
"""PyTorch Dataset for ChoirSMT training.

Loads (image, token_sequence) pairs from a manifest file.
The image is the rendered+augmented page; the labels are the tokenized MusicXML.
"""
from __future__ import annotations
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

IMAGE_SIZE = (1024, 1536)  # height x width — preserves aspect, fits A100 memory


class ChoirOMRDataset(Dataset):
    """Pairs of (image, token_ids).

    Expects a manifest JSON file with entries:
      [{"image": "path/to/page.webp", "tokens": [4, 17, 32, ...]}, ...]

    Token IDs are integers from the vocabulary built by tokenize_satb.build_vocab().
    """
    def __init__(self, manifest_path: Path, image_root: Path | None = None,
                 image_size: tuple[int, int] = IMAGE_SIZE, max_seq_len: int = 4096):
        self.entries = json.loads(Path(manifest_path).read_text())
        self.image_root = Path(image_root) if image_root else Path(manifest_path).parent
        self.image_size = image_size
        self.max_seq_len = max_seq_len

        self.transform = T.Compose([
            T.Grayscale(num_output_channels=3),  # 3 channels so ConvNeXt accepts it
            T.Resize(image_size, antialias=True),
            T.ToTensor(),  # 0..1 float
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
        ])

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        img_path = self.image_root / e["image"]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        tokens = e["tokens"][:self.max_seq_len]
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        return img_tensor, token_tensor


def collate_pad(batch):
    """Pads token sequences to the longest in the batch."""
    imgs, toks = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    max_len = max(t.size(0) for t in toks)
    padded = torch.zeros(len(toks), max_len, dtype=torch.long)  # pad_id = 0
    for i, t in enumerate(toks):
        padded[i, :t.size(0)] = t
    return imgs, padded
