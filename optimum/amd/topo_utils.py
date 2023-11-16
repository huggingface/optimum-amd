from itertools import combinations

import amdsmi
import numpy as np


def get_bandwidth_matrix():
    amdsmi.amdsmi_init()
    devices = amdsmi.amdsmi_get_device_handles()

    num_devices = len(devices)
    bandwidth_matrix = [[None for _ in range(num_devices)] for _ in range(num_devices)]

    # direct bandwidth
    for i, src_device in enumerate(devices):
        for j, dst_device in enumerate(devices):
            if i == j:
                bandwidth_matrix[i][j] = float("inf")
            else:
                try:
                    curr_bandwidth = amdsmi.amdsmi_get_minmax_bandwidth(src_device, dst_device)["max_bandwidth"]
                    if curr_bandwidth != 0:
                        bandwidth_matrix[i][j] = curr_bandwidth
                except Exception:
                    pass

    # indirect bandwidth
    for i in range(num_devices):
        for j in range(num_devices):
            if bandwidth_matrix[i][j] is None:
                maxmin_bandwidth = 0
                for k in range(num_devices):
                    if k == i or k == j:
                        continue
                    elif bandwidth_matrix[i][k] is not None and bandwidth_matrix[k][j] is not None:
                        min_bandwidth = min(bandwidth_matrix[i][k], bandwidth_matrix[k][j])
                        if min_bandwidth > maxmin_bandwidth:
                            bandwidth_matrix[i][j] = min_bandwidth
                            maxmin_bandwidth = min_bandwidth

    # fill missing values
    for i in range(num_devices):
        for j in range(num_devices):
            if bandwidth_matrix[i][j] is None:
                bandwidth_matrix[i][j] = 0

    amdsmi.amdsmi_shut_down()

    return bandwidth_matrix


def extract_max_avg_bandwidth_cluster(bandwidth_matrix, cluster_num_devices):
    if len(bandwidth_matrix) < cluster_num_devices:
        raise ValueError("Number of devices in the cluster cannot be greater than the number of devices in the system")

    if cluster_num_devices == 1:
        return [0], float("inf")

    num_devices = range(len(bandwidth_matrix))

    max_avg_bandwidth = 0
    max_bandwidth_cluster = None

    for cluster in combinations(num_devices, cluster_num_devices):
        curr_bandwidth_matrix = [[bandwidth_matrix[i][j] for i in cluster] for j in cluster]
        curr_avg_bandwidth = np.mean(curr_bandwidth_matrix, where=~np.eye(len(curr_bandwidth_matrix), dtype=bool))
        if curr_avg_bandwidth > max_avg_bandwidth:
            max_avg_bandwidth = curr_avg_bandwidth
            max_bandwidth_cluster = list(cluster)

    return max_bandwidth_cluster, max_avg_bandwidth


def extract_min_avg_bandwidth_cluster(bandwidth_matrix, cluster_num_devices):
    if len(bandwidth_matrix) < cluster_num_devices:
        raise ValueError("Number of devices in the cluster cannot be greater than the number of devices in the system")

    if cluster_num_devices == 1:
        return [0], float("inf")

    num_devices = range(len(bandwidth_matrix))

    min_avg_bandwidth = float("inf")
    min_bandwidth_cluster = None

    for cluster in combinations(num_devices, cluster_num_devices):
        curr_bandwidth_matrix = [[bandwidth_matrix[i][j] for i in cluster] for j in cluster]
        curr_avg_bandwidth = np.mean(curr_bandwidth_matrix, where=~np.eye(len(curr_bandwidth_matrix), dtype=bool))
        if curr_avg_bandwidth < min_avg_bandwidth:
            min_avg_bandwidth = curr_avg_bandwidth
            min_bandwidth_cluster = list(cluster)

    return min_bandwidth_cluster, min_avg_bandwidth
