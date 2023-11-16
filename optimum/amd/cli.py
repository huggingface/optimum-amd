import os
import sys
from argparse import ArgumentParser
from .topo_utils import get_bandwidth_matrix, extract_max_avg_bandwidth_cluster, extract_min_avg_bandwidth_cluster


def amdrun():
    assert len(sys.argv) >= 4, "Usage: python amdrun.py --ngpus <num_gpus> <script> <script_args>"
    # parse the script and script args
    script, *script_args = sys.argv[3:]
    # remove the script and script args
    sys.argv = sys.argv[:3]

    parser = ArgumentParser()
    parser.add_argument(
        "--ngpus", "--nproc-per-node", type=int, default=2, help="Number of devices used in the experiment"
    )

    args = parser.parse_args()
    ngpus = args.ngpus

    bandwidth_matrix = get_bandwidth_matrix()
    max_avg_bandwidth_cluster, max_avg_bandwidth = extract_max_avg_bandwidth_cluster(bandwidth_matrix, ngpus)

    print(f"MaxAvg NUMA bandwidth cluster: {max_avg_bandwidth_cluster}")
    print(f"MaxAvg NUMA bandwidth: {max_avg_bandwidth}")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(map(str, max_avg_bandwidth_cluster)))

    # run the script
    os.system(f"python {script} {' '.join(script_args)}")


def wrongrun():
    assert len(sys.argv) >= 4, "Usage: python amdrun.py --ngpus <num_gpus> <script> <script_args>"
    script, *script_args = sys.argv[3:]
    sys.argv = sys.argv[:3]

    parser = ArgumentParser()
    parser.add_argument(
        "--ngpus", "--nproc-per-node", type=int, default=2, help="Number of devices used in the experiment"
    )

    args = parser.parse_args()
    ngpus = args.ngpus

    bandwidth_matrix = get_bandwidth_matrix()
    min_avg_bandwidth_cluster, min_avg_bandwidth = extract_min_avg_bandwidth_cluster(bandwidth_matrix, ngpus)

    print(f"MinAvg NUMA bandwidth cluster: {min_avg_bandwidth_cluster}")
    print(f"MinAvg NUMA bandwidth: {min_avg_bandwidth}")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(map(str, min_avg_bandwidth_cluster)))

    # run the script
    if len(script_args) > 0:
        os.system(f"python {script} {' '.join(script_args)}")
    else:
        os.system(f"python {script}")
