from pathlib import Path
import submitit
import sys

from training.cfm_mnist import train_cfm_mnist, sample_class

handle = """
        Usage: main.py <stage> [class label]

        Stage:
            train - train conditional flow matching Unet on MNIST
            sample - sample a class with trained model

        Class label (optional):
            0-9 - specify MNIST class to generate an image for

        Examples:
            python3 launch_cfm_mnist.py train
            python3 launch_cfm_mnist.py sample 8
        """

def main(log_folder, job_name, stage="train", class_label=None):

    folder = log_folder
    job_name = job_name
    env = Path(sys.executable).parent.parent
    act_env = f"source {env}/bin/activate"

    # Setup SLURM executor
    executor = submitit.AutoExecutor(folder=folder)
    executor.update_parameters(
        slurm_job_name=job_name,
        timeout_min=2880,  # 2 days
        mem_gb=50,
        slurm_setup=[act_env],
        slurm_stderr_to_stdout=True,
        slurm_additional_parameters={
            "gres": "gpu:3g.40gb:1",
            # "output": f"{folder}mnist_test_%j.out",
            # "error": f"{folder}mnist_test_%j.err",
            },
        cpus_per_task=32,
        )

    # Submit job with arguments (if required)
    if stage == "train":
        job = executor.submit(train_cfm_mnist)
    elif stage == "sample":
        job = executor.submit(sample_class, class_label)
    else:
        print("Please enter either a valid stage (train|sample)")
    print("Submitted job ID:", job.job_id)

if __name__ == "__main__":
    try:
        stage = sys.argv[1]
        if stage == "train":
            main("test_logs/", "test_mnist", stage="train")
        elif stage == "sample":
            try:
                label = int(sys.argv[2])
                assert 0 <= label <= 9, f"Class index must be between 0 and 9, got {label}"
                main("test_logs/", "test_mnist", stage="sample", class_label=label)
            except IndexError:
                print(handle)
        else:
            print("Please enter either a valid stage (train|sample)")
    except IndexError:
        print("Error: Missing args!")
        print(handle)
        sys.exit(1)