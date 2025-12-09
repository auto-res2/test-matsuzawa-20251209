# src/preprocess.py
# Complete preprocessing pipeline for specified datasets (Caltech101, Oxford-IIIT Pet)
import os
from pathlib import Path
from typing import Tuple
from torchvision import transforms
from torchvision.datasets import Caltech101, OxfordIIITPet

# This module provides dataset loading and standard transforms for the experiments.
# All datasets cached under .cache/

CACHE_DIR = Path(".cache/")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_image_transforms(image_size: int = 224, center_crop: bool = True, normalize_stats: Tuple[float, float, float] = None):
    if normalize_stats is None:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean, std = normalize_stats

    tlist = []
    tlist.append(transforms.Resize(image_size))
    if center_crop:
        tlist.append(transforms.CenterCrop(image_size))
    tlist.append(transforms.ToTensor())
    tlist.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(tlist)


def load_caltech101(root: str = ".cache/", image_size: int = 224, center_crop: bool = True):
    transform = get_image_transforms(image_size, center_crop)
    ds = Caltech101(root=root, download=True, transform=transform)
    return ds


def load_oxford_pets(root: str = ".cache/", image_size: int = 224, center_crop: bool = True):
    transform = get_image_transforms(image_size, center_crop)
    ds = OxfordIIITPet(root=root, download=True, transform=transform, target_types='category')
    return ds
