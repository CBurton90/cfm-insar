from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler
from torchvision.utils import make_grid
from tqdm import tqdm

from flow_matching.sampler import PathSampler
from flow_matching.solver import ModelWrapper, ODESolver
from models.unet import UNetModelWrapper
from utils.TenGeoP_SARwv_webdataset import create_dataloader

def train_cfm():

    batch_size=48
    epochs=40
    num_workers=1
    sigma_min=0.0
    lr=1e-3

    print(f"Cuda is available: {torch.cuda.is_available()}")
    device = torch.device("cuda")
    
    trainloader, train_list = create_dataloader(
        url="https://www.seanoe.org/data/00456/56796/data/58684.tar.gz",
        info_url="https://www.seanoe.org/data/00456/56796/data/58683.txt",
        mode='train',
        batch_size=batch_size,
        num_workers=num_workers,
        cache_path="/share/home/conradb/.cache/webdataset"
        )
    
    valloader, val_list = create_dataloader(
        url="https://www.seanoe.org/data/00456/56796/data/58684.tar.gz",
        info_url="https://www.seanoe.org/data/00456/56796/data/58683.txt",
        mode='val',
        batch_size=batch_size,
        num_workers=num_workers,
        cache_path="/share/home/conradb/.cache/webdataset"
        )

    # print("-"*20)
    # print("train len: ", len(train_list))
    # print("-"*20)
    # print("val len: ", len(val_list))
    # print(train_list & val_list)

    # for i, sample in enumerate(trainloader):
    #     if i > 100:
    #         break
    #     print(sample[0].shape)
    #     print(sample[0].min(), sample[0].max())
    #     print(sample[1])

    input_shape = next(iter(trainloader))[0][0].size() # TenGeoP_SARwv is 1x256x256 as we have used torches RandomCrop

    flow = UNetModelWrapper(
        input_shape,
        num_channels=64,
        num_res_blocks=2,
        num_classes=10,
        class_cond=True,
    ).to(device)

    path_sampler = PathSampler(sigma_min=sigma_min)

    # Load the optimizer
    optimizer = torch.optim.AdamW(flow.parameters(), lr=lr)
    scaler = GradScaler(enabled=device.type == "cuda")

    for epoch in range(epochs):
        
        flow.train()
        tracked_loss = 0
        count = 0

        for step, (x_1, y, _) in enumerate(tqdm(trainloader, total=len(train_list)//batch_size, mininterval=60*2)):
            x_1, y = x_1.to(device), y.to(device)
            # Compute the probability path samples
            x_0 = torch.randn_like(x_1)
            t = torch.rand(x_1.size(0), device=device, dtype=x_1.dtype)
            x_t, dx_t = path_sampler.sample(x_0, x_1, t)

            flow.zero_grad(set_to_none=True)

            # Compute the conditional flow matching loss with class conditioning
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                vf_t = flow(t=t, x=x_t, y=y)
                loss = F.mse_loss(vf_t, dx_t)
            
            tracked_loss += loss.item()
            count += step    

            # Gradient scaling and backprop
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)  # clip gradients
            scaler.step(optimizer)
            scaler.update()

        print(f"Epoch {epoch} complete, train loss is: {tracked_loss / count}")

    output_dir = Path(__file__).parent.parent / "checkpoints" / "cfm" / "TenGeoP_SARwv"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(flow.state_dict(), output_dir / f"TenGeoP_SARwv_ckpt_{epochs}epochs.pth")

def sample_class(class_label):

    epochs = 40

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flow = UNetModelWrapper(
        (1,256,256),
        num_channels=64,
        num_res_blocks=2,
        num_classes=10,
        class_cond=True,
    ).to(device)

    output_dir = Path(__file__).parent.parent / "checkpoints" / "cfm" / "TenGeoP_SARwv"

    state_dict = torch.load(output_dir / f"TenGeoP_SARwv_ckpt_{epochs}epochs.pth", map_location=device, weights_only=True)
    flow.load_state_dict(state_dict)
    flow.eval()

    # Use ODE solver to sample trajectories
    class WrappedModel(ModelWrapper):
        def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
            return self.model(x=x, t=t, **extras)

    samples_per_class = 4
    sample_steps = 101
    time_steps = torch.linspace(0, 1, sample_steps).to(device)
    class_list = torch.arange(10, device=device).repeat(samples_per_class)

    wrapped_model = WrappedModel(flow)
    step_size = 0.05
    x_init = torch.randn((class_list.size(0), 1, 256, 256), dtype=torch.float32, device=device)
    solver = ODESolver(wrapped_model)
    sol = solver.sample(
        x_init=x_init,
        step_size=step_size,
        method="midpoint",
        time_grid=time_steps,
        return_intermediates=False,
        y=class_list,
    )
    sol = sol.detach().cpu()
    print(sol.shape)
    # final_samples = sol[-1]

    # from torchvision.utils import save_image
    # # Denormalize from [-1, 1] to [0, 1]
    # batch_denorm = (sol + 1) / 2
    # save_image(batch_denorm, 'sar_output.png')

    fig, ax = plt.subplots(figsize=(15,10))
    grid = make_grid((sol+1)/2, nrow=10, normalize=False)
    ax.imshow(grid.permute(1, 2, 0))
    ax.set_title("Final samples (t = 1.0)", fontsize=16)
    ax.axis("off")
    fig.savefig("sar_grid.png", dpi=300)