from pathlib import Path
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler
from tqdm import tqdm

from flow_matching.sampler import PathSampler
from flow_matching.solver import ModelWrapper, ODESolver
from models.unet import UNetModelWrapper
from utils.TenGeoP_SARwv_webdataset import create_dataloader

def train_cfm_mnist():

    batch_size=64
    epochs=10
    num_workers=1
    sigma_min=0.0
    lr=1e-3

    print(f"Cuda is available: {torch.cuda.is_available()}")
    device = torch.device("cuda")
    
    trainloader = create_dataloader(
        url="https://www.seanoe.org/data/00456/56796/data/58684.tar.gz",
        mode='train',
        batch_size=batch_size,
        num_workers=num_workers
        )