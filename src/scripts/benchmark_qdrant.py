#!/usr/bin/env python3
"""
Benchmark: Measure Qdrant retrieval performance.

Metrics:
- Query latency (min, max, avg)
- Throughput (queries/sec)
"""

import json
import os
import time
from statistics import mean, stdev

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointIdsList


def benchmark():
    """Benchmark Qdrant performance."""

    print("\n" + "=" * 70)
    print("BENCHMARK: Qdrant Performance")
    print("=" * 70)

    # Load embeddings
    with open("data/embeddings.json", "r") as f:
        embeddings = json.load(f)

    # Connect
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    # Warm up
    print("\n🔥 Warming up (5 queries)...")
    for i in range(5):
        client.query_points(
            collection_name="regulations",
            query=embeddings[0]["embedding"],
            limit=10,
        )

    # Benchmark: 100 queries with different embeddings
    print("\n⏱️  Benchmarking (100 queries)...")
    latencies = []
    for i in range(100):
        start = time.time()
        client.query_points(
            collection_name="regulations",
            query=embeddings[i % len(embeddings)]["embedding"],
            limit=10,
        )
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)
        if (i + 1) % 20 == 0:
            print(f"   {i + 1}/100 completed")

    # Statistics
    min_latency = min(latencies)
    max_latency = max(latencies)
    avg_latency = mean(latencies)
    std_latency = stdev(latencies) if len(latencies) > 1 else 0
    throughput = 1000 / avg_latency  # queries/sec

    print(f"\n📊 Results (100 queries):")
    print(f"   Min latency:  {min_latency:.2f} ms")
    print(f"   Max latency:  {max_latency:.2f} ms")
    print(f"   Avg latency:  {avg_latency:.2f} ms")
    print(f"   Std dev:      {std_latency:.2f} ms")
    print(f"   Throughput:   {throughput:.1f} queries/sec")

    print("\n" + "=" * 70)
    print("✅ BENCHMARK COMPLETE")
    print("=" * 70 + "\n")

    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump({
            "queries": 100,
            "min_ms": round(min_latency, 2),
            "max_ms": round(max_latency, 2),
            "avg_ms": round(avg_latency, 2),
            "std_ms": round(std_latency, 2),
            "throughput_qps": round(throughput, 1),
        }, f, indent=2)

    print("📈 Results saved to benchmark_results.json\n")


if __name__ == "__main__":
    benchmark()