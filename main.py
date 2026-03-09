from pathlib import Path
import submitit
import sys

from training.cfm_TenGeoP_SARwv import train_cfm, sample_class

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
            # "gres": "gpu:3g.40gb:1",
            # "output": f"{folder}mnist_test_%j.out",
            # "error": f"{folder}mnist_test_%j.err",
            },
        cpus_per_task=32,
        )

    # Submit job with arguments (if required)
    # job = executor.submit(train_cfm)
    job = executor.submit(sample_class, 5)
    print("Submitted job ID:", job.job_id)


if __name__ == "__main__":
    main("test_logs/", "test_TenGeoP-SAR", stage="train")
