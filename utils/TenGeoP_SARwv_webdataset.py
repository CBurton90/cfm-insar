import io
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms_v2
import webdataset as wds






def create_dataloader(url="https://www.seanoe.org/data/00456/56796/data/58684.tar.gz", mode='train', batch_size=64, num_workers=1):
    
    """
    Create a webdataset dataloader for the TenGeoP-SARwv dataset:
    
    Wang Chen, Mouche Alexis, Tandeo Pierre, Stopa Justin, Longépé Nicolas, Erhard Guillaume, Foster Ralph, Vandemark Douglas, Chapron Bertrand (2018).
    Labeled SAR imagery dataset of ten geophysical phenomena from Sentinel-1 wave mode (TenGeoP-SARwv). SEANOE. https://doi.org/10.17882/56796

    We offer class balanced train/val/test splits replicating:

    Tuel, A., Kerdreux, T., Hulbert, C., & Rouet-Leduc, B. (2023). Diffusion models for interferometric satellite aperture radar. arXiv preprint arXiv:2308.16847.
    """
    
    # url = f"pipe:aws s3 cp {s3_bucket} - {flag}"
    url = "https://www.seanoe.org/data/00456/56796/data/58684.tar.gz"
        
    dataset = (
        wds.WebDataset(
            url,
            cache_dir=None,
            shardshuffle=False,
            resampled=False,
            )
            .select(lambda s: s['__key__'])
            # .select(lambda s: s['__key__'].startswith(f"mnist_png/{'training' if mode=='train' else 'testing'}/"))
            # .shuffle(60000, initial=60000) # Shuffle samples for those that are preloaded with initial, MNIST is small so lets preload all and shuffle all
            # # .decode("pil") # This will map to RGB, MNIST is grayscale, if used the grayscale torch transform should be implemented 
            # .decode(gs_decode) # Custom decode function to open PIL image in grayscale
            # .map(lambda s: {**s, 'cls': int(s['__key__'].split('/')[2])})
            # .to_tuple("png", "cls")
            # .map(torch_transform)
            # .batched(batch_size, partial=False) # no partial batches
    )

    # loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers)
    # loader = loader.unbatched().shuffle(10).batched(batch_size, partial=False) # shuffle amongst workers

    for i, sample in enumerate(dataset):
        if i > 100:
            break
        print(sample)

    # return loader