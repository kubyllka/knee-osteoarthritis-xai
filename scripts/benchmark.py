import argparse
import statistics
from pathlib import Path
from time import perf_counter

import requests


def benchmark(api_url: str, image_path: Path, requests_count: int) -> None:
    latencies = []
    started = perf_counter()

    for _ in range(requests_count):
        with image_path.open("rb") as image_file:
            response = requests.post(
                f"{api_url.rstrip('/')}/predict",
                files={"file": (image_path.name, image_file, "image/jpeg")},
                timeout=30,
            )
        response.raise_for_status()
        latencies.append(response.json()["latency_ms"])

    total_seconds = perf_counter() - started
    throughput = requests_count / total_seconds

    print("Benchmark results")
    print(f"Requests: {requests_count}")
    print(f"Total time: {total_seconds:.3f} s")
    print(f"Throughput: {throughput:.2f} req/s")
    print(f"Mean latency: {statistics.mean(latencies):.2f} ms")
    print(f"Median latency: {statistics.median(latencies):.2f} ms")
    print(f"Min latency: {min(latencies):.2f} ms")
    print(f"Max latency: {max(latencies):.2f} ms")
    if len(latencies) >= 20:
        p95 = statistics.quantiles(latencies, n=20)[18]
        print(f"P95 latency: {p95:.2f} ms")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark the FastAPI ONNX inference endpoint.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark(args.api_url, args.image, args.requests)
