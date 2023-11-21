import torch


def synchronize():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(f"cuda:{i}")


def benchmark(i, j):
    tensor = torch.randn(10000, 10000, device=f"cuda:{i}")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    synchronize()
    start_event.record()
    synchronize()
    tensor.to(f"cuda:{j}")
    synchronize()
    end_event.record()
    synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    return elapsed_time_ms


def main():
    latencies = {}
    for i in range(torch.cuda.device_count()):
        for j in range(torch.cuda.device_count()):
            if i == j:
                continue

            # warmup
            for k in range(10):
                benchmark(i, j)

            # benchmark
            for k in range(10):
                if (i, j) not in latencies:
                    latencies[(i, j)] = []

                latencies[(i, j)].append(benchmark(i, j))

            print(f"Transfer Latency from {i} to {j} is {sum(latencies[(i, j)]) / len(latencies[(i, j)])} ms")

    print(
        f"Average Transfer Latency is {sum([sum(latencies[(i, j)]) / len(latencies[(i, j)]) for i, j in latencies]) / len(latencies)} ms"
    )


if __name__ == "__main__":
    import os

    LOCAL_RANK = os.environ.get("LOCAL_RANK", None)

    if LOCAL_RANK is None or LOCAL_RANK == "0":
        main()
