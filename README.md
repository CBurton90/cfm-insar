# cfm-insar
Conditional Flow Matching (Lipman et al., 2023) for InSAR

## Usage

### MNIST

Train a simple conditional flow matching Unet model on MNIST. MNIST data is streamed from FastAI's AWS open data S3 bucket with webdataset.

Training is setup for a SLURM HPC cluster with a Nvidia MIG parition, if you are using a standard Nvidia gpu you can change the following line in `launch_cfm_mnist.py`: 

`slurm_additional_parameters={"gres": "gpu:3g.40gb:1"}`

To run training for 10 epochs:

`uv run launch_cfm_mnist.py train`

To run sampling:

`uv run launch_cfm_mnist.py sample 7` (this will generate a sample conditioned on MNIST class 7)