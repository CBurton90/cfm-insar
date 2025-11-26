import io
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms_v2
import webdataset as wds

def torch_transform(sample):

    """Use torchvision transforms in the webdataset pipeline"""

    transforms = transforms_v2.Compose([
        transforms_v2.ToImage(), # Convert PIL to uint8 tensor between 0-255
        # transforms_v2.Grayscale(num_output_channels=1), # Only uncomment if MNIST has been converted to RGB via the standard webdataset .decode('pil')
        transforms_v2.ToDtype(torch.float32, scale=True), # Convert to float32 and scale between 0-1
        transforms_v2.RandomHorizontalFlip(),
        transforms_v2.Normalize((0.5,), (0.5,)) # Normalize between -1-1, common in diffusion
    ])

    image, label = sample

    return transforms(image), label

def gs_decode(key, sample):

    """Custom decoder for grayscale PIL images"""

    if key.endswith(('.png', '.PNG', '.jpg', '.jpeg', '.JPG', '.JPEG')):
        return Image.open(io.BytesIO(sample)).convert('L')

def create_dataloader(s3_bucket="s3://fast-ai-imageclas/mnist_png.tgz", open_data=True, mode='train', batch_size=64, num_workers=1):
    
    """Create a webdataset dataloader for MNIST uploaded by FastAI on AWS S3"""
    
    if open_data:
        flag = "--no-sign-request"
        url = f"pipe:aws s3 cp {s3_bucket} - {flag}"
        # url = "pipe:s3cmd get s3://fast-ai-imageclas/mnist_png.tgz -"
    else:
        url = f"pipe:aws s3 cp {s3_bucket} -"
        
    dataset = (
        wds.WebDataset(
            url,
            cache_dir=None,
            shardshuffle=False,
            resampled=False,
            )
            .select(lambda s: s['__key__'].startswith(f"mnist_png/{'training' if mode=='train' else 'testing'}/"))
            .shuffle(60000, initial=60000) # Shuffle samples for those that are preloaded with initial, MNIST is small so lets preload all and shuffle all
            # .decode("pil") # This will map to RGB, MNIST is grayscale, if used the grayscale torch transform should be implemented 
            .decode(gs_decode) # Custom decode function to open PIL image in grayscale
            .map(lambda s: {**s, 'cls': int(s['__key__'].split('/')[2])})
            .to_tuple("png", "cls")
            .map(torch_transform)
            .batched(batch_size, partial=False) # no partial batches
    )

    loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers)
    loader = loader.unbatched().shuffle(10).batched(batch_size, partial=False) # shuffle amongst workers

    return loader