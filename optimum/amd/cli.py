import os
import subprocess
import sys


try:
    subprocess.run(["rocm-smi"], check=True)

    ROCM_AVAILABLE = True
except Exception:
    ROCM_AVAILABLE = False

try:
    import zentorch  # noqa: F401 # type: ignore

    ZENTORCH_AVAILABLE = True
except Exception:
    ZENTORCH_AVAILABLE = False


def get_amd_zentorch_env():
    """
    A cli command that sets a couple of zentorch-optimal environment variables
    before running the command passed to it.

    Usage: amdrun <script> <script_args>
    Example: amdrun torchrun --nproc_per_node 4 train.py
    """

    CPU_COUNT = os.cpu_count()
    MALLOC_CONF = "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
    LD_PRELOAD = "/usr/lib/x86_64-linux-gnu/libjemalloc.so"
    GOMP_CPU_AFFINITY = f"0-{CPU_COUNT - 1}"
    OMP_NUM_THREADS = f"{CPU_COUNT}"
    OMP_WAIT_POLICY = "ACTIVE"
    ZENDNN_GEMM_ALGO = "4"
    OMP_DYNAMIC = "False"

    env = {}

    env["OMP_NUM_THREADS"] = OMP_NUM_THREADS
    env["OMP_DYNAMIC"] = OMP_DYNAMIC
    env["OMP_WAIT_POLICY"] = OMP_WAIT_POLICY
    env["ZENDNN_GEMM_ALGO"] = ZENDNN_GEMM_ALGO
    env["GOMP_CPU_AFFINITY"] = GOMP_CPU_AFFINITY
    env["LD_PRELOAD"] = LD_PRELOAD
    env["MALLOC_CONF"] = MALLOC_CONF

    return env


def get_amd_rocm_env():
    """
    A cli command that sets the target GPUs to use based on the maximum average bandwidth
    between them.

    Usage: amdrun <script> <script_args>
    Example: amdrun torchrun --nproc_per_node 4 train.py
    """

    from .topology_utils import extract_max_avg_bandwidth_cluster, get_bandwidth_matrix

    # extract the number of devices to use
    if "--nproc_per_node" in sys.argv:
        # torchrun style
        nproc_per_node_index = sys.argv.index("--nproc_per_node")
        num_devices = int(sys.argv[nproc_per_node_index + 1])
    elif "--ngpus" in sys.argv:
        # accelerate/deepspeed style
        ngpus_index = sys.argv.index("--ngpus")
        num_devices = int(sys.argv[ngpus_index + 1])
    else:
        # early exit if we can't find the number of devices
        return {}

    bandwidth_matrix = get_bandwidth_matrix()
    max_avg_bandwidth_cluster, max_avg_bandwidth = extract_max_avg_bandwidth_cluster(bandwidth_matrix, num_devices)

    # lowest level isolation env var on AMD GPUs
    ROCR_VISIBLE_DEVICES = ",".join(list(map(str, max_avg_bandwidth_cluster)))

    print(f"MaxAvg NUMA bandwidth cluster: {max_avg_bandwidth_cluster}")
    print(f"MaxAvg NUMA bandwidth: {max_avg_bandwidth}")
    print(f"ROCR_VISIBLE_DEVICES: {ROCR_VISIBLE_DEVICES}")

    return {"ROCR_VISIBLE_DEVICES": ROCR_VISIBLE_DEVICES}


def amdrun():
    """
    A cli command that sets a couple of zentorch & rocm environment variables to maximize
    performance.

    Usage: amdrun <script> <script_args>
    Example: amdrun torchrun --nproc_per_node 4 train.py
    """
    env = {}

    if ROCM_AVAILABLE:
        env.update(get_amd_rocm_env())

    if ZENTORCH_AVAILABLE:
        env.update(get_amd_zentorch_env())

    args = [f"{k}={v}" for k, v in env.items()] + sys.argv[1:]
    exit_code = subprocess.run(args, shell=True).returncode
    sys.exit(exit_code)
