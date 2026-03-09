from PIL import Image
import numpy as np
import os
import random
from rasterio.io import MemoryFile
import requests
import tarfile
import torch
import torchvision.transforms.v2 as transforms_v2
import webdataset as wds

_transforms = transforms_v2.Compose([
        transforms_v2.RandomCrop(256),
        transforms_v2.ToImage(), # Convert PIL to uint8 tensor between 0-255
        transforms_v2.ToDtype(torch.float32, scale=True), # Convert to float32 and scale between 0-1
        transforms_v2.RandomHorizontalFlip(),
        transforms_v2.RandomVerticalFlip(),
        transforms_v2.Normalize((0.5,), (0.5,)) # Normalize between -1-1, common in diffusion
    ])

def torch_transform(image):

    """Use torchvision transforms in the webdataset pipeline"""

    return _transforms(image)

def build_splits(text_url, label_map, mode="train", seed=42):

    random.seed(seed)
    response = requests.get(text_url)
    metadata = response.text.strip().split('\n')

    file_names = []
    for key, value in label_map.items():
        grouped = []
        for f in metadata:
            f_id = f.strip().split()[0]
            if f_id.startswith(f"{key}/"):
                grouped.append(f_id.split(".")[0])
        
        random.shuffle(grouped)
        n_train = int(len(grouped) * 0.8)
        n_val = int(len(grouped) * 0.05)
 
        if mode == "train":
            split = grouped[:n_train]
        elif mode == "val":
            split = grouped[n_train:n_train + n_val]
        else:
            split = grouped[n_train + n_val:]

        file_names.extend(split)
    random.shuffle(file_names)

    return file_names


def gs_decode(key, sample):

    """Custom decoder for TIFF to grayscale PIL images"""

    if key.endswith(('.tiff', '.TIFF', '.tif', '.TIF')):

        # Read GeoTIFF
        with MemoryFile(sample) as memfile:
            with memfile.open() as src:
                # There is only one band
                try:
                    data = src.read(1)
                except:
                    print(f"GeoTIFF {key} could not be read")

        # Normalise to [0, 255], this will not preserve relative amplitudes between SAR images (per sample norm)
        data_norm = 255. * (data - np.min(data)) / (np.max(data) - np.min(data))

        return Image.fromarray(data_norm).convert('L')


def create_dataloader(url="https://www.seanoe.org/data/00456/56796/data/58684.tar.gz", info_url="https://www.seanoe.org/data/00456/56796/data/58683.txt", mode='train', batch_size=64, num_workers=1, cache_path ="/share/home/conradb/.cache/webdataset"):
    
    """
    Create a webdataset dataloader for the TenGeoP-SARwv dataset:
    
    Wang Chen, Mouche Alexis, Tandeo Pierre, Stopa Justin, Longépé Nicolas, Erhard Guillaume, Foster Ralph, Vandemark Douglas, Chapron Bertrand (2018).
    Labeled SAR imagery dataset of ten geophysical phenomena from Sentinel-1 wave mode (TenGeoP-SARwv). SEANOE. https://doi.org/10.17882/56796

    We offer class balanced train/val/test splits replicating:

    Tuel, A., Kerdreux, T., Hulbert, C., & Rouet-Leduc, B. (2023). Diffusion models for interferometric satellite aperture radar. arXiv preprint arXiv:2308.16847.
    """
    
    # url = f"pipe:aws s3 cp {s3_bucket} - {flag}"
    url = "https://www.seanoe.org/data/00456/56796/data/58684.tar.gz"

    label_map = {'F': 0, 'G': 1, 'H': 2, 'I': 3, 'J': 4, 'K': 5, 'L': 6, 'M': 7, 'N': 8, 'O': 9}

    flist = build_splits(info_url, label_map, mode=mode, seed=42)
    fset = set(flist)
    os.makedirs(cache_path, exist_ok=True)
        
    dataset = (
        wds.WebDataset(
            url,
            cache_dir=cache_path,
            shardshuffle=False,
            resampled=False,
            )
            # .select(lambda s: s['__key__'].startswith("GeoTIFF"))
            .select(lambda s: s['__key__'].replace("GeoTIFF/", "") in fset)
            .shuffle(size=400, initial=400)
            .decode(gs_decode)
            .map(lambda s: {**s, 'cls': int(label_map[s['__key__'].split('/')[1]])})
            .to_tuple("tiff", "cls", "__key__")
            .map_tuple(torch_transform)
            .batched(batch_size, partial=False)
            )

    loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers)
    # loader = loader.unbatched().shuffle(100).batched(batch_size, partial=False) # shuffle amongst workers

    return loader, fset