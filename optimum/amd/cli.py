import os
import sys
from argparse import ArgumentParser

from .topo_utils import extract_max_avg_bandwidth_cluster, get_bandwidth_matrix


def amdrun():
    """
    An alternative to torchrun that's optimized to maximize inter-devices bandwidth.

    Usage: amdrun --ngpus <num_gpus> <script> <script_args>
    """

    # 4 is the minimum number of arguments required (amdrun --ngpus <num_gpus> <script>)
    assert len(sys.argv) >= 4, "Usage: amdrun --ngpus <num_gpus> <script> <script_args>"

    script, *script_args = sys.argv[3:]
    sys.argv = sys.argv[:3]

    parser = ArgumentParser()
    parser.add_argument(
        "--nproc_per_node",
        "--ngpus",
        type=int,
        default=2,
        help="Number of processes to run per node or equivalently the number of GPUs to use",
    )

    args = parser.parse_args()
    ngpus = args.ngpus

    bandwidth_matrix = get_bandwidth_matrix()
    max_avg_bandwidth_cluster, max_avg_bandwidth = extract_max_avg_bandwidth_cluster(bandwidth_matrix, ngpus)

    print(f"MaxAvg NUMA bandwidth cluster: {max_avg_bandwidth_cluster}")
    print(f"MaxAvg NUMA bandwidth: {max_avg_bandwidth}")

    CUDA_VISIBLE_DEVICES = ",".join(list(map(str, max_avg_bandwidth_cluster)))

    # run the script
    if len(script_args) > 0:
        os.system(
            f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} torchrun --nproc_per_node={ngpus} {script} {' '.join(script_args)}"
        )
    else:
        os.system(f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} torchrun --nproc_per_node={ngpus} {script}")
