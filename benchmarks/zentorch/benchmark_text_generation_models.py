from optimum_benchmark.backends.pytorch.config import PyTorchConfig
from optimum_benchmark.benchmarks.inference.config import InferenceConfig
from optimum_benchmark.experiment import ExperimentConfig, launch
from optimum_benchmark.launchers.process.config import ProcessConfig
from optimum_benchmark.logging_utils import setup_logging


setup_logging(level="ERROR")

REPO_ID = "optimum-amd/ci-benchmarks"
EXPERIMENT_NAME = "text_generation_models/static_cache"

# for list with static cache support
# https://github.com/search?q=repo%3Ahuggingface%2Ftransformers+_setup_cache%28self&type=code
STATIC_CACHE_MODELS_LIST = [
    "google/gemma-2b",
    "google/gemma-7b",
    "huggyllama/llama-7b",
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
]
INPUT_SHAPES = {
    "batch_size": 1,
    "sequence_length": 2,
}
GENERATE_KWARGS = {
    "max_new_tokens": 2,
    "min_new_tokens": 2,
}
TORCH_COMPILE_CONFIG = {
    "backend": "zentorch",
}
CACHE_IMPLEMENTATION = "static"


def benchmark_text_generation_with_static_cache():
    for model in STATIC_CACHE_MODELS_LIST:
        launcher_config = ProcessConfig(start_method="spawn")  # isolated process
        benchmark_config = InferenceConfig(
            memory=True,
            latency=True,
            input_shapes=INPUT_SHAPES,
            generate_kwargs=GENERATE_KWARGS,
        )
        backend_config = PyTorchConfig(
            model=model,
            device="cpu",
            no_weights=True,
            torch_compile=True,
            torch_compile_config=TORCH_COMPILE_CONFIG,
            cache_implementation=CACHE_IMPLEMENTATION,
        )

        experiment_config = ExperimentConfig(
            experiment_name=EXPERIMENT_NAME,
            benchmark=benchmark_config,
            launcher=launcher_config,
            backend=backend_config,
        )

        benchmark_report = launch(experiment_config)

        # experiment_config.push_to_hub(
        #     save_path=f"zentorch/{EXPERIMENT_NAME}/{model}",
        #     commit_message="Added experiment config",
        #     repo_id=REPO_ID,
        #     private=True,
        # )
        # benchmark_report.push_to_hub(
        #     save_path=f"zentorch/{EXPERIMENT_NAME}/{model}",
        #     commit_message="Added benchmark report",
        #     repo_id=REPO_ID,
        #     private=True,
        # )


if __name__ == "__main__":
    benchmark_text_generation_with_static_cache()
