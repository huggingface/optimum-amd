import os
import torch
import psutil
from optimum_benchmark import Benchmark, BenchmarkConfig, InferenceConfig, ProcessConfig, PyTorchConfig
import json
from huggingface_hub import hf_hub_download

torch._dynamo.reset()


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
    version,
    repo_id,
    cache_implementation,
):
    BENCHMARK_NAME = (
        f"benchmark_epyc_{device}_{backend}_dtype_{dtype}_single_instance/{version}/"
        f"{model.replace('/', '_')}/"
        f"cores_{num_cores}_instances_{num_instances}/"
        f"batch_{batch_size}_prompt_{sequence_length}_gen_{decode_length}/instance_{instance}"
    )

    benchmark_names = []
    for i in range(num_instances):
        benchmark_names.append(
            f"benchmark_epyc_{device}_{backend}_dtype_{dtype}_single_instance/{version}/"
            f"{model.replace('/', '_')}/"
            f"cores_{num_cores}_instances_{num_instances}/"
            f"batch_{batch_size}_prompt_{sequence_length}_gen_{decode_length}/instance_{i}"
        )

    benchmark_report_path = None
    try:
        for benchmark_name in benchmark_names:
            benchmark_report = os.path.join(benchmark_name, "benchmark_report.json")
            benchmark_report_path = hf_hub_download(repo_id=repo_id, filename=benchmark_report, repo_type="dataset")
            with open(benchmark_report_path, "r") as f:
                report = json.load(f)
            with open("benchmark_info.log", "a") as f:
                f.write(f"Found {benchmark_report}\n")
    except Exception as e:
        benchmark_report_path = None
        with open("benchmark_info.log", "a") as f:
            f.write(f"Not Found {e}\n")

    if benchmark_report_path is not None:
        return

    result = f"Model: {model}, Backend: {backend}, Batch Size: {batch_size}, Sequence Length: {sequence_length}, Decode Length: {decode_length}, Num instances: {num_instances} and and Instance: {instance}, membind {numactl_kwargs['membind']}, Device: {device}, Instance: {instance}, Num Instances: {num_instances}, Num Cores: {num_cores}"

    with open("benchmark_info.log", "a") as f:
        f.write(f"Running benchmark for {result}\n")

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
            cache_implementation=cache_implementation,
        )

        benchmark_config = BenchmarkConfig(
            name=BENCHMARK_NAME, launcher=launcher_config, scenario=scenario_config, backend=backend_config
        )

        benchmark_report = Benchmark.launch(benchmark_config)
        benchmark_config.push_to_hub(
            commit_message=f"Added {result}",
            subfolder=BENCHMARK_NAME,
            repo_id=repo_id,
            private=True,
        )
        benchmark_report.push_to_hub(
            commit_message=f"Added {result}",
            subfolder=BENCHMARK_NAME,
            repo_id=repo_id,
            private=True,
        )
    except Exception as e:
        print(f"Failed to run {result}", flush=True)
        print(e, flush=True)

        with open("benchmark_error.log", "a") as f:
            f.write(f"Failed to {result} {str(e)}\n")


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
    parser.add_argument("--num_cores", type=int, help="Num cores", required=False, default=None)
    parser.add_argument("--version", type=str, help="Zendnn library version", required=False, default="5_rc7")
    parser.add_argument("--repo_id", type=str, help="Repo id to upload benchmark", required=True)
    parser.add_argument("--cache_implementation", type=str, help="Cache implementation", required=True)
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
    num_cores_given = args.num_cores
    version = args.version
    repo_id = args.repo_id
    cache_implementation = args.cache_implementation

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

    if num_cores_given:
        os.environ["OMP_NUM_THREADS"] = str(num_cores_given)
        num_cores = num_cores_given
    else:
        os.environ["OMP_NUM_THREADS"] = str(num_cores)

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
        version=version,
        repo_id=repo_id,
        cache_implementation=cache_implementation,
    )
