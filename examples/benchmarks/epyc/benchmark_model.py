import os
import torch
from optimum_benchmark import Benchmark, BenchmarkConfig, InferenceConfig, ProcessConfig, PyTorchConfig

def argparser():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark models")
    parser.add_argument("--physcpubind", type=str, help="Physical CPU binding", required=True)
    parser.add_argument("--membind", type=int, help="Memory binding", required=True)
    parser.add_argument("--model_id", type=str, help="Model ID", required=True)
    return parser.parse_args()

REPO_ID = "optimum-amd/zendnn-benchmarks"
torch._dynamo.reset()
# for list with static cache support
# https://github.com/search?q=repo%3Ahuggingface%2Ftransformers+_setup_cache%28self&type=code
MODELS_DECODER = [
    "google/gemma-2-9b-it",
    "EleutherAI/gpt-j-6B",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen1.5-14B-Chat",
]

STATIC_CACHE_MODELS = [
    "google/gemma-2-9b-it",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

INPUT_SHAPES = {
    "batch_size": 1,
    "sequence_length": 1920,
}
GENERATE_KWARGS = {
    "max_new_tokens": 128,
    "min_new_tokens": 128,
}

def benchmark(phycpubind_str, membind, model_id):
    task = "text-generation"
    for dtype in ["bfloat16"]:
        for backend in ["zentorch"]:
            for model in [model_id]:
                print(f"Running benchmark for {model} with dtype {dtype} and backend {backend}")
                launcher_config = ProcessConfig(
                    start_method="spawn",
                    numactl=True,
                    numactl_kwargs={
                        "cpunodebind": membind,
                        "membind": membind,
                        "physcpubind": phycpubind_str,
                    },
                )  # isolated process
                scenario_config = InferenceConfig(
                    memory=True,
                    latency=True,
                    input_shapes=INPUT_SHAPES,
                    generate_kwargs=GENERATE_KWARGS,
                    iterations=3,
                    warmup_runs=2,
                )

                try:
                    backend_config = PyTorchConfig(
                        model=model,
                        device="cpu",
                        no_weights=True,
                        torch_compile=True,
                        torch_compile_target="forward",
                        torch_compile_config={"backend": backend,},
                        task="text-generation",
                        torch_dtype="bfloat16",
                        cache_implementation="static" if model in STATIC_CACHE_MODELS else None,
                    )
                    
                    bs = INPUT_SHAPES["batch_size"]
                    sl = INPUT_SHAPES["sequence_length"]
                    maxt = GENERATE_KWARGS["max_new_tokens"]

                    BENCHMARK_NAME = f"benchmark_epyc_turin_{backend}_multi_instance/dtype_{dtype}/{task}/batch_{bs}_cores_8_instances_64/batch_{bs}_prompt_{sl}_gen_{maxt}_cores_{phycpubind_str}"
                    subfolder = f"{BENCHMARK_NAME}/{model.replace('/', '_')}"

                    benchmark_config = BenchmarkConfig(
                        name=BENCHMARK_NAME,
                        launcher=launcher_config,
                        scenario=scenario_config,
                        backend=backend_config
                    )

                    benchmark_report = Benchmark.launch(benchmark_config)

                    benchmark_config.push_to_hub(
                        commit_message="Added benchmark config",
                        subfolder=subfolder,
                        repo_id=REPO_ID,
                        private=True,
                    )
                    benchmark_report.push_to_hub(
                        commit_message="Added benchmark report",
                        subfolder=subfolder,
                        repo_id=REPO_ID,
                        private=True,
                    )
                except Exception as e:
                    print(f"Failed to run benchmark for {model} with dtype {dtype} and backend {backend}")
                    print(e)
                    continue

if __name__ == "__main__":
    args = argparser()
    phycpubind = f"{args.physcpubind}"
    membind = int(args.membind)
    model_id = args.model_id
    print(f"Running benchmarks for models with CPU binding {phycpubind} and memory binding {membind}")
    benchmark(phycpubind, membind, model_id)
