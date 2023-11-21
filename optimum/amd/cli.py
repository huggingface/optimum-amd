import os
import sys
from argparse import ArgumentParser

from .topo_utils import extract_max_avg_bandwidth_cluster, extract_min_avg_bandwidth_cluster, get_bandwidth_matrix


def amdrun():
    """
    An alternative to torchrun that's optimized to maximize NUMA bandwidth.

    Usage: amdrun --ngpus <num_gpus> <script> <script_args>
    """

    # 4 is the minimum number of arguments required (amdrun --ngpus <num_gpus> <script>)
    assert len(sys.argv) >= 4, "Usage: amdrun --ngpus <num_gpus> <script> <script_args>"

    script, *script_args = sys.argv[3:]
    sys.argv = sys.argv[:3]

    parser = ArgumentParser()
    parser.add_argument(
        "--ngpus", "--nproc_per_node", type=int, default=2, help="Number of devices used in the experiment"
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


def wrongrun():
    """
    An alternative to torchrun that's optimized to minimize NUMA bandwidth.

    Usage: wrongrun --ngpus <num_gpus> <script> <script_args>
    """

    # 4 is the minimum number of arguments required (wrongrun --ngpus <num_gpus> <script>)
    assert len(sys.argv) >= 4, "Usage: wrongrun --ngpus <num_gpus> <script> <script_args>"

    script, *script_args = sys.argv[3:]
    sys.argv = sys.argv[:3]

    parser = ArgumentParser()
    parser.add_argument(
        "--ngpus", "--nproc_per_node", type=int, default=2, help="Number of devices used in the experiment"
    )

    args = parser.parse_args()
    ngpus = args.ngpus

    bandwidth_matrix = get_bandwidth_matrix()
    min_avg_bandwidth_cluster, min_avg_bandwidth = extract_min_avg_bandwidth_cluster(bandwidth_matrix, ngpus)

    print(f"MinAvg NUMA bandwidth cluster: {min_avg_bandwidth_cluster}")
    print(f"MinAvg NUMA bandwidth: {min_avg_bandwidth}")

    CUDA_VISIBLE_DEVICES = ",".join(list(map(str, min_avg_bandwidth_cluster)))

    # run the script
    if len(script_args) > 0:
        os.system(
            f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} torchrun --nproc_per_node={ngpus} {script} {' '.join(script_args)}"
        )
    else:
        os.system(f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} torchrun --nproc_per_node={ngpus} {script}")
