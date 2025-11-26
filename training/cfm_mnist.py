from pathlib import Path
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler
from tqdm import tqdm

from flow_matching.sampler import PathSampler
from flow_matching.solver import ModelWrapper, ODESolver
from models.unet import UNetModelWrapper
from utils.mnist_webdataset import create_dataloader

def train_cfm_mnist():

    batch_size=64
    epochs=10
    num_workers=4
    sigma_min=0.0
    lr=1e-3

    print(f"Cuda is available: {torch.cuda.is_available()}")
    device = torch.device("cuda")

    trainloader = create_dataloader(
        s3_bucket="s3://fast-ai-imageclas/mnist_png.tgz",
        open_data=True,
        mode='train',
        batch_size=batch_size,
        num_workers=num_workers
        )

    input_shape = next(iter(trainloader))[0][0].size() # MNIST is 1x28x28

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

        for x_1, y in tqdm(trainloader):
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

            # Gradient scaling and backprop
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)  # clip gradients
            scaler.step(optimizer)
            scaler.update()

    output_dir = Path(__file__).parent.parent / "checkpoints" / "cfm" / "mnist"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(flow.state_dict(), output_dir / f"mnist_ckpt_{epochs}epochs.pth")

def sample_class(class_label):

    epochs=10

    device = torch.device("cuda")

    flow = UNetModelWrapper(
        (1,28,28),
        num_channels=64,
        num_res_blocks=2,
        num_classes=10,
        class_cond=True,
    ).to(device)

    output_dir = Path(__file__).parent.parent / "checkpoints" / "cfm" / "mnist"

    state_dict = torch.load(output_dir / f"mnist_ckpt_{epochs}epochs.pth", weights_only=True)
    flow.load_state_dict(state_dict)
    flow.eval()

    # Use ODE solver to sample trajectories
    class WrappedModel(ModelWrapper):
        def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
            return self.model(x=x, t=t, **extras)

    samples_per_class = 1
    sample_steps = 101
    time_steps = torch.linspace(0, 1, sample_steps).to(device)
    class_list = torch.arange(class_label,class_label+1, device=device).repeat(samples_per_class)

    wrapped_model = WrappedModel(flow)
    step_size = 0.05
    x_init = torch.randn((class_list.size(0), 1, 28, 28), dtype=torch.float32, device=device)
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

    from torchvision.utils import save_image
    # Denormalize from [-1, 1] to [0, 1]
    batch_denorm = (sol + 1) / 2
    save_image(batch_denorm, 'output.png')
