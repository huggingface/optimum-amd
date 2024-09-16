import os
import torch
import psutil
from optimum_benchmark import Benchmark, BenchmarkConfig, InferenceConfig, ProcessConfig, PyTorchConfig

# for list with static cache support
# https://github.com/search?q=repo%3Ahuggingface%2Ftransformers+_setup_cache%28self&type=code
# MODELS_DECODER = [
#     "google/gemma-2-9b-it",
#     "EleutherAI/gpt-j-6B",
#     "meta-llama/Llama-2-7b-chat-hf",
#     "meta-llama/Llama-2-13b-chat-hf",
#     "meta-llama/Meta-Llama-3-8B-Instruct",
#     "mistralai/Mistral-7B-Instruct-v0.3",
#     "Qwen/Qwen2-7B-Instruct",
#     "Qwen/Qwen1.5-14B-Chat",
# ]

REPO_ID = "optimum-amd/zendnn-benchmarks"
torch._dynamo.reset()

STATIC_CACHE_MODELS = [
    "google/gemma-2-9b-it",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


version = "5_rc"


def benchmark(
    model,
    task,
    dtype,
    backend,
    batch_size,
    sequence_length,
    decode_length,
    numactl_kwargs,
    device,
    instance,
    num_instances,
    num_cores,
):
    BENCHMARK_NAME = (
        f"benchmark_epyc_{device}_{backend}_dtype_{dtype}_multi_instance/{version}/"
        f"{model.replace('/', '_')}/"
        f"cores_{num_cores}_instances_{num_instances}/"
        f"batch_{batch_size}_prompt_{sequence_length}_gen_{decode_length}/instance_{instance}"
    )
    
    with open("benchmarkxx.log", "a") as f:
        f.write(f"Running benchmark for {model} with dtype {dtype} and backend {backend} Num instances: {num_instances} and and Instance: {instance}\n")

    launcher_config = ProcessConfig(
        start_method="spawn",
        numactl=True,
        numactl_kwargs=numactl_kwargs,
    )  # isolated process
    scenario_config = InferenceConfig(
        memory=True,
        latency=True,
        input_shapes={
            "batch_size": batch_size,
            "sequence_length": sequence_length,
        },
        generate_kwargs={
            "max_new_tokens": decode_length,
            "min_new_tokens": decode_length,
        },
        iterations=3,
        warmup_runs=2,
    )

    try:
        backend_config = PyTorchConfig(
            model=model,
            device="cpu",
            no_weights=False,
            torch_compile=True,
            torch_compile_target="forward",
            torch_compile_config={
                "backend": backend,
            },
            task=task,
            torch_dtype=dtype,
            cache_implementation="static" if model in STATIC_CACHE_MODELS else None,
        )

        benchmark_config = BenchmarkConfig(
            name=BENCHMARK_NAME, launcher=launcher_config, scenario=scenario_config, backend=backend_config
        )

        benchmark_report = Benchmark.launch(benchmark_config)
        benchmark_config.push_to_hub(
            commit_message=f"Added benchmark config {model} with batch size {batch_size} and sequence length {sequence_length}",
            subfolder=BENCHMARK_NAME,
            repo_id=REPO_ID,
            private=True,
        )
        benchmark_report.push_to_hub(
            commit_message=f"Added benchmark report {model} with batch size {batch_size} and sequence length {sequence_length}",
            subfolder=BENCHMARK_NAME,
            repo_id=REPO_ID,
            private=True,
        )
    except Exception as e:
        print(f"Failed to run benchmark for {model} with dtype {dtype} and backend {backend}", flush=True)
        print(e, flush=True)

        with open("benchmark_error.log", "a") as f:
            f.write(f"Failed to run benchmark for {model} with dtype {dtype} and backend {backend} and task {task}\n")
            f.write(str(e))


def argparser():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark models")
    parser.add_argument("--physcpubind", type=str, help="Physical CPU binding", default=None)
    parser.add_argument("--membind", type=int, help="Memory binding", required=True)
    parser.add_argument("--model_id", type=str, help="Model ID", required=True)
    parser.add_argument("--batch_size", type=int, help="Sequence Length", required=True)
    parser.add_argument("--sequence_length", type=int, help="Sequence Length", required=True)
    parser.add_argument("--decode_length", type=int, help="Decode Length", required=True)
    parser.add_argument("--backend", type=str, help="Backend", required=True)
    parser.add_argument("--dtype", type=str, help="Data type", default="bfloat16")
    parser.add_argument("--task", type=str, help="Task", default="text-generation")
    parser.add_argument("--device", type=str, help="Device", default="turin")
    parser.add_argument("--num_instances", type=int, help="Number of instances", required=True)
    parser.add_argument("--instance", type=int, help="Instance", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()

    phycpubind = args.physcpubind
    membind = int(args.membind)
    model = args.model_id
    sequence_length = int(args.sequence_length)
    decode_length = int(args.decode_length)
    batch_size = int(args.batch_size)
    backend = args.backend
    dtype = args.dtype
    task = args.task
    device = args.device
    num_instances = args.num_instances
    instance = args.instance

    numactl_kwargs = {
        "cpunodebind": membind,
        "membind": membind,
    }
    if phycpubind:
        numactl_kwargs["physcpubind"] = phycpubind

    physical_cores = psutil.cpu_count(logical=False)
    logical_cpus = psutil.cpu_count(logical=True)
    threads_per_core = logical_cpus // physical_cores
    num_cores = physical_cores // num_instances
    os.environ["OMP_NUM_THREADS"] = str(num_cores*threads_per_core)

    # print(f"Running benchmark for {model} with dtype {dtype} and backend {backend} and task {task}")
    # print(f"Batch size: {batch_size}")
    # print(f"Sequence length: {sequence_length}")
    # print(f"Decode length: {decode_length}")
    # print(f"Numactl kwargs: {numactl_kwargs}")
    # print(f"Device: {device}")
    # print(f"Instance: {instance}")
    # print(f"Num instances: {num_instances}")
    # print(f"Num cores: {num_cores}")

    benchmark(
        model=model,
        task=task,
        dtype=dtype,
        backend=backend,
        batch_size=batch_size,
        sequence_length=sequence_length,
        decode_length=decode_length,
        numactl_kwargs=numactl_kwargs,
        device=device,
        instance=instance,
        num_instances=num_instances,
        num_cores=num_cores,
    )
