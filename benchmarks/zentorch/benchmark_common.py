from optimum_benchmark.backends.pytorch.config import PyTorchConfig
from optimum_benchmark.benchmarks.inference.config import InferenceConfig
from optimum_benchmark.experiment import ExperimentConfig, launch
from optimum_benchmark.launchers.process.config import ProcessConfig


REPO_ID = "optimum-amd/ci-benchmarks"
EXPERIMENT_NAME = "zentorch_common"

MODELS_LIST = [
    "google-bert/bert-base-uncased",
]
INPUT_SHAPES = {
    "batch_size": 1,
    "sequence_length": 2,
}
TORCH_COMPILE_CONFIG = {
    "backend": "zentorch",
}


def benchmark_common():
    for model in MODELS_LIST:
        launcher_config = ProcessConfig(start_method="spawn")  # isolated process
        benchmark_config = InferenceConfig(
            memory=True,
            latency=True,
            input_shapes=INPUT_SHAPES,
        )
        backend_config = PyTorchConfig(
            model=model,
            device="cpu",
            no_weights=True,
            torch_compile=True,
            torch_compile_config=TORCH_COMPILE_CONFIG,
        )

        experiment_config = ExperimentConfig(
            experiment_name=EXPERIMENT_NAME,
            benchmark=benchmark_config,
            launcher=launcher_config,
            backend=backend_config,
        )

        benchmark_report = launch(experiment_config)

        experiment_config.push_to_hub(
            commit_message="Added experiment config",
            subfolder=f"{EXPERIMENT_NAME}/{model}",
            repo_id=REPO_ID,
            private=True,
        )
        benchmark_report.push_to_hub(
            commit_message="Added benchmark report",
            subfolder=f"{EXPERIMENT_NAME}/{model}",
            repo_id=REPO_ID,
            private=True,
        )


if __name__ == "__main__":
    benchmark_common()
