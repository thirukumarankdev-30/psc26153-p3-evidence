#!/usr/bin/env python3
"""
latency_test.py - HTTP Latency & Throughput Benchmark for Attack Onset API.

Team: TECH ROOKIES, PSC26153 (II-year Engineering Students)
Target: Python 3.10.12, httpx, psutil, pandas, numpy.

Purpose:
Sends sequential HTTP POST requests to the /predict endpoint using evaluation window
feature vectors. Measures end-to-end HTTP request/response latency (p50, p95, p99),
monitors system and server process CPU utilization, and outputs latency_log.csv
and latency_summary.json.
"""

import argparse
import datetime
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
import psutil

CANONICAL_FEATURES = [
    "flow_count",
    "syn_rate",
    "total_fwd_packets",
    "total_bwd_packets",
    "mean_flow_duration",
    "mean_packet_length",
    "unique_src_count",
    "unique_dst_count",
    "mean_total_packets",
]

# Aliases mapping raw column names back to canonical feature names
FEATURE_ALIASES = {
    "flow_count": ["flow_count", "Flow Count", "flows"],
    "syn_rate": ["syn_rate", "SYN Rate", "syn_flag_rate"],
    "total_fwd_packets": ["total_fwd_packets", "Total Fwd Packets", "fwd_packets"],
    "total_bwd_packets": ["total_bwd_packets", "Total Backward Packets", "bwd_packets"],
    "mean_flow_duration": ["mean_flow_duration", "Mean Flow Duration", "avg_flow_duration"],
    "mean_packet_length": ["mean_packet_length", "Mean Packet Length", "avg_packet_length"],
    "unique_src_count": ["unique_src_count", "Unique Source Count", "unique_sources"],
    "unique_dst_count": ["unique_dst_count", "Unique Destination Count", "unique_destinations"],
    "mean_total_packets": ["mean_total_packets", "Mean Total Packets", "avg_total_packets"],
}


def load_and_map_features(source_path: str) -> pd.DataFrame:
    """
    Reads the feature CSV, maps any aliases to canonical names, and verifies
    that all 9 canonical features are present.

    Fails explicitly if the file is a *_log.csv or lacks feature columns.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Feature source file does not exist: {source_path}")

    print(f"[*] Reading feature rows from {source_path}...")
    df = pd.read_csv(source_path)
    clean_cols = {col.strip(): col for col in df.columns}

    # Check for common mistake: supplying a log.csv instead of an eval_features.csv
    log_markers = {"true_label", "episode_id", "score", "alarm"}
    if log_markers.issubset(set(clean_cols.keys())) and "flow_count" not in clean_cols:
        raise ValueError(
            f"Input file '{source_path}' appears to be a harness log CSV (*_log.csv), "
            "not a features file! Please provide an eval features file (e.g. R01_eval_features.csv)."
        )

    # Resolve aliases
    rename_dict = {}
    missing = []
    for canonical, aliases in FEATURE_ALIASES.items():
        matched = None
        for alias in aliases:
            if alias in clean_cols:
                matched = clean_cols[alias]
                break
        if matched is not None:
            rename_dict[matched] = canonical
        else:
            missing.append(canonical)

    if missing:
        raise ValueError(
            f"Source file '{source_path}' is missing canonical feature column(s): {missing}.\n"
            f"Found columns: {list(clean_cols.keys())}"
        )

    canonical_df = df.rename(columns=rename_dict)[CANONICAL_FEATURES].copy()
    print(f"[*] Successfully validated and mapped {len(canonical_df):,} feature rows.")
    return canonical_df


def main():
    parser = argparse.ArgumentParser(
        description="Latency benchmark tool for containerised Attack Onset Forecast API"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of timed requests to send (default: 1000)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="R01_eval_features.csv",
        help="Path to evaluation features CSV (default: R01_eval_features.csv)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL of the serving endpoint (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of initial untimed warmup requests (default: 20)",
    )
    parser.add_argument(
        "--server-pid",
        type=int,
        default=None,
        help="Optional OS PID of the server process to monitor CPU usage",
    )

    args = parser.parse_args()

    # Load canonical feature dataset
    features_df = load_and_map_features(args.source)
    n_rows = len(features_df)

    if n_rows < args.n:
        print(
            f"\n[!] WARNING: Source file has {n_rows} rows, which is fewer than requested --n {args.n}.\n"
            f"    Will run exactly {n_rows} timed requests without synthetic replication.\n"
        )
        total_requests = n_rows
    else:
        total_requests = args.n

    # Set up server process monitoring if PID was supplied
    server_process = None
    if args.server_pid is not None:
        try:
            server_process = psutil.Process(args.server_pid)
            print(f"[*] Attached CPU monitor to server PID {args.server_pid} ({server_process.name()}).")
        except psutil.NoSuchProcess:
            print(f"[!] Warning: Process with PID {args.server_pid} was not found.")
        except Exception as exc:
            print(f"[!] Warning: Could not attach to PID {args.server_pid}: {exc}")

    print(
        "[*] Note: If the server is running inside a Docker container on Windows or macOS, "
        "the container process runs inside a virtual machine and may not be visible by host PID."
    )

    predict_url = f"{args.base_url.rstrip('/')}/predict"
    health_url = f"{args.base_url.rstrip('/')}/health"

    # CPU sampling collections
    system_cpu_samples = []
    server_cpu_samples = []

    # Prime psutil CPU monitoring
    psutil.cpu_percent(interval=None)
    if server_process:
        try:
            server_process.cpu_percent(interval=None)
        except Exception:
            pass

    # Use HTTP client with persistent connection pool
    with httpx.Client(timeout=30.0) as client:
        # Check endpoint connectivity
        try:
            h_resp = client.get(health_url)
            if h_resp.status_code != 200:
                print(f"[!] Warning: Health check returned HTTP {h_resp.status_code}: {h_resp.text}")
        except Exception as exc:
            print(f"[!] Error: Unable to connect to {health_url}: {exc}")
            print("    Please ensure the API server or container is running before testing.")
            sys.exit(1)

        # Warmup phase (excluded from latency and failure stats)
        if args.warmup > 0 and n_rows > 0:
            print(f"[*] Sending {args.warmup} warmup request(s)...")
            warmup_row = features_df.iloc[0].to_dict()
            for _ in range(args.warmup):
                try:
                    client.post(predict_url, json={"features": warmup_row})
                except Exception:
                    pass

        # Timed benchmark loop
        print(f"[*] Commencing benchmark: sending {total_requests} sequential timed request(s)...")
        log_records = []
        latencies_ms = []
        failed_count = 0

        # Sample CPU initially
        system_cpu_samples.append(psutil.cpu_percent(interval=None))
        if server_process:
            try:
                server_cpu_samples.append(server_process.cpu_percent(interval=None))
            except Exception:
                pass

        for idx in range(total_requests):
            row_dict = features_df.iloc[idx].to_dict()
            payload = {"features": row_dict}

            t0 = time.perf_counter()
            status_code = 0
            is_failed = 0
            err_msg = ""

            try:
                resp = client.post(predict_url, json=payload)
                t1 = time.perf_counter()
                status_code = resp.status_code
                if status_code == 200:
                    _ = resp.json()
                else:
                    is_failed = 1
                    err_msg = f"HTTP {status_code}: {resp.text[:100]}"
            except Exception as exc:
                t1 = time.perf_counter()
                status_code = 500
                is_failed = 1
                err_msg = str(exc)

            lat_ms = (t1 - t0) * 1000.0
            latencies_ms.append(lat_ms)

            if is_failed:
                failed_count += 1

            log_records.append({
                "request_idx": idx + 1,
                "latency_ms": round(lat_ms, 3),
                "status_code": status_code,
                "failed": is_failed,
                "error": err_msg,
            })

            # Sample CPU periodically (every 100 requests)
            if (idx + 1) % 100 == 0:
                system_cpu_samples.append(psutil.cpu_percent(interval=None))
                if server_process:
                    try:
                        server_cpu_samples.append(server_process.cpu_percent(interval=None))
                    except Exception:
                        pass

        # Final CPU sample
        system_cpu_samples.append(psutil.cpu_percent(interval=None))
        if server_process:
            try:
                server_cpu_samples.append(server_process.cpu_percent(interval=None))
            except Exception:
                pass

    # Compute percentiles
    p50_lat = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    p95_lat = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
    p99_lat = float(np.percentile(latencies_ms, 99)) if latencies_ms else 0.0

    mean_sys_cpu = float(np.mean(system_cpu_samples)) if system_cpu_samples else 0.0
    mean_srv_cpu = float(np.mean(server_cpu_samples)) if server_cpu_samples else None

    # Write latency_log.csv
    log_df = pd.DataFrame(log_records)
    log_df.to_csv("latency_log.csv", index=False)
    print(f"[*] Per-request latency log saved to latency_log.csv ({len(log_df)} records)")

    # Write latency_summary.json
    summary_data = {
        "n_requests": total_requests,
        "failed": failed_count,
        "p50_ms": round(p50_lat, 3),
        "p95_ms": round(p95_lat, 3),
        "p99_ms": round(p99_lat, 3),
        "mean_system_cpu_pct": round(mean_sys_cpu, 2),
        "mean_server_cpu_pct": round(mean_srv_cpu, 2) if mean_srv_cpu is not None else None,
        "source": args.source,
        "base_url": args.base_url,
        "timestamp_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    with open("latency_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print("[*] Benchmark summary saved to latency_summary.json")

    # Console summary output
    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK RESULTS:")
    print(f"Requests Sent : {total_requests}")
    print(f"Failed Count  : {failed_count}")
    print(f"p50 Latency   : {p50_lat:.3f} ms")
    print(f"p95 Latency   : {p95_lat:.3f} ms")
    print(f"p99 Latency   : {p99_lat:.3f} ms")
    print(f"Mean System CPU: {mean_sys_cpu:.2f}%")
    if mean_srv_cpu is not None:
        print(f"Mean Server CPU: {mean_srv_cpu:.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
