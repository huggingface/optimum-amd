import logging
import os
import subprocess
import sys


logger = logging.getLogger(__name__)

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


def get_env_vars_overrides():
    """
    Returns a dictionary of environment variables that are set in the command line arguments.
    """

    env = {}

    for arg in sys.argv:
        if "=" in arg:
            key, value = arg.split("=")
            env[key] = value

    return env


def get_amd_zentorch_env():
    """
    Returns a dictionary of environment variables that are optimized for the AMD ZenTorch plugin.

    The target environment variables are:
    - `OMP_NUM_THREADS`: The number of OpenMP threads to use.
    - `OMP_DYNAMIC`: Whether or not OpenMP threads are dynamically allocated.
    - `OMP_WAIT_POLICY`: The OpenMP wait policy.
    - `ZENDNN_GEMM_ALGO`: The GEMM algorithm to use.
    - `GOMP_CPU_AFFINITY`: The CPU affinity for OpenMP threads.
    - `LD_PRELOAD`: The path to the Jemalloc library.
    - `MALLOC_CONF`: The Jemalloc configuration.
    """

    # TODO: how to handle NUMA nodes and socket affinity?
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

    if not os.path.exists(LD_PRELOAD) and "LD_PRELOAD" not in get_env_vars_overrides():
        logger.warning(
            f"Jemalloc not found at {LD_PRELOAD} either because it's not installed or because the path is incorrect."
            "Make sure it's installed and/or override `LD_PRELOAD` manually: `amdrun LD_PRELOAD=/path/to/libjemalloc.so python script.py script_args`"
        )

    logger.info("AMD ZenTorch environment variables:")
    logger.info(f"- OMP_NUM_THREADS: {OMP_NUM_THREADS}")
    logger.info(f"- OMP_DYNAMIC: {OMP_DYNAMIC}")
    logger.info(f"- OMP_WAIT_POLICY: {OMP_WAIT_POLICY}")
    logger.info(f"- ZENDNN_GEMM_ALGO: {ZENDNN_GEMM_ALGO}")
    logger.info(f"- GOMP_CPU_AFFINITY: {GOMP_CPU_AFFINITY}")
    logger.info(f"- LD_PRELOAD: {LD_PRELOAD}")
    logger.info(f"- MALLOC_CONF: {MALLOC_CONF}")

    return env


def get_amd_rocm_env():
    """
    Returns a dictionary of environment variables that are optimized for AMD's ROCm platform.

    The target environment variables are:
    - `ROCR_VISIBLE_DEVICES`: The list of devices to use (maximizing the average bandwidth between them).
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

    logger.info("AMD ROCm environment variables:")
    logger.info(f"- ROCR_VISIBLE_DEVICES: {ROCR_VISIBLE_DEVICES}")

    return {"ROCR_VISIBLE_DEVICES": ROCR_VISIBLE_DEVICES}


def amdrun():
    """
    A cli command that sets a couple of ZenTorch & ROCm environment variables to maximize performance.

    Usage: amdrun <script> <script_args>
    Example: amdrun torchrun --nproc_per_node 4 train.py
    """
    env = os.environ.copy()

    if ROCM_AVAILABLE:
        env.update(get_amd_rocm_env())

    if ZENTORCH_AVAILABLE:
        env.update(get_amd_zentorch_env())

    exit_code = subprocess.run(sys.argv[1:], env=env).returncode

    sys.exit(exit_code)
