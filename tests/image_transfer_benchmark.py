#!/usr/bin/env python3
"""
6G-RESCUE Edge Benchmarking Script — SCP Transfer
===================================================
Transfers images to the edge node via SCP (SSH) instead of HTTP.
Steps through batch sizes: 1, 5, 10, 20, 50, 100, 200, 500, 1000
For each batch size N, transfers N images and records TOTAL transfer time.

Dependencies:
    pip install paramiko

Usage:
    python benchmark_edge.py --label no_delay
    python benchmark_edge.py --label delay_100ms
    python benchmark_edge.py --image-dir /path/to/coco/val2017 --label loss_10pct
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import paramiko
except ImportError:
    sys.exit("[ERROR] Missing dependency — run:  pip install paramiko")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (edit these to match your setup)
# ──────────────────────────────────────────────────────────────────────────────
HOST         = "22scomps001.ncl.ac.uk"
PORT         = 22
USERNAME     = "akumar"
PASSWORD     = "jetson"          # ← set your password here
REMOTE_DIR   = "/home/akumar/images"      # Destination folder on edge server

IMAGE_DIR    = "./val2017"
TIMEOUT_S    = 200                       # SSH connect + transfer timeout (keep above tc delay)

BATCH_SIZES  = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]

SUPPORTED    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="6G-RESCUE SCP benchmark — total transfer time per batch")
    p.add_argument("--host",       default=HOST)
    p.add_argument("--port",       type=int, default=PORT)
    p.add_argument("--username",   default=USERNAME)
    p.add_argument("--password",   default=PASSWORD)
    p.add_argument("--remote-dir", default=REMOTE_DIR)
    p.add_argument("--image-dir",  default=IMAGE_DIR)
    p.add_argument("--label",      default="run",
                   help="Label for this tc profile, e.g. 'no_delay', 'delay_100ms'")
    return p.parse_args()


def load_images(image_dir: str) -> list:
    folder = Path(image_dir)
    if not folder.exists():
        sys.exit(f"[ERROR] Directory not found: {folder.resolve()}")
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED)
    if not images:
        sys.exit(f"[ERROR] No supported images in {folder.resolve()}")
    if len(images) < max(BATCH_SIZES):
        sys.exit(f"[ERROR] Need at least {max(BATCH_SIZES)} images, found {len(images)}.")
    print(f"[INFO] {len(images)} images found in '{folder}'.")
    return images


def open_ssh(args) -> paramiko.SSHClient:
    """Open and return an authenticated SSH connection."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs = dict(
        hostname=args.host,
        port=args.port,
        username=args.username,
        timeout=TIMEOUT_S,
    )
    connect_kwargs["password"] = args.password
    client.connect(**connect_kwargs)
    return client


def ensure_remote_dir(ssh: paramiko.SSHClient, remote_dir: str):
    """Create destination directory on edge if it doesn't exist."""
    ssh.exec_command(f"mkdir -p {remote_dir}")


def run_batch(args, images: list, batch_size: int) -> dict:
    """
    Transfer `batch_size` images via SCP.
    Opens one SSH connection for the whole batch.
    Returns result dict with total transfer time.
    """
    subset     = images[:batch_size]
    successful = 0
    failed     = 0
    errors     = []

    print(f"\n  Transferring {batch_size} image(s) via SCP...", end=" ", flush=True)

    t_start = time.perf_counter()
    try:
        ssh  = open_ssh(args)
        ensure_remote_dir(ssh, args.remote_dir)
        sftp = ssh.open_sftp()

        for image_path in subset:
            remote_path = f"{args.remote_dir}/{image_path.name}"
            try:
                sftp.put(str(image_path), remote_path)
                successful += 1
            except Exception as e:
                failed += 1
                err = str(e)[:60]
                if err not in errors:
                    errors.append(err)

        sftp.close()
        ssh.close()

    except paramiko.AuthenticationException:
        errors.append("AUTH_FAILED")
        failed = batch_size
    except paramiko.SSHException as e:
        errors.append(f"SSH_ERR: {str(e)[:60]}")
        failed = batch_size
    except Exception as e:
        errors.append(f"ERROR: {str(e)[:60]}")
        failed = batch_size

    total_s    = round(time.perf_counter() - t_start, 4)
    total_mb   = round(sum(p.stat().st_size for p in subset) / (1024 * 1024), 4)

    status = "\033[92m✓\033[0m" if failed == 0 else f"\033[91m✗ {failed} failed\033[0m"
    print(f"{status}  total: {total_s:.3f}s  |  {total_mb:.2f} MB"
          + (f"  errors: {', '.join(errors[:3])}" if errors else ""))

    return {
        "label":           None,        # filled in main
        "batch_size":      batch_size,
        "successful":      successful,
        "failed":          failed,
        "total_time_s":    total_s,
        "avg_per_img_s":   round(total_s / batch_size, 4),
        "total_data_mb":   total_mb,
        "throughput_mb_s": round(total_mb / total_s, 4) if total_s > 0 else 0.0,
        "errors":          "; ".join(errors) if errors else "",
        "timestamp_utc":   datetime.utcnow().isoformat(),
    }


def write_csv(results: list, label: str) -> str:
    path   = f"benchmark_{label}.csv"
    fields = ["label", "batch_size", "successful", "failed",
              "total_time_s", "avg_per_img_s",
              "total_data_mb", "throughput_mb_s",
              "errors", "timestamp_utc"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    return path


def print_summary(results: list, label: str, csv_path: str, host: str):
    W = 85
    print("\n\n" + "═" * W)
    print(f"  6G-RESCUE SCP BENCHMARK — {label.upper()}")
    print("═" * W)
    print(f"  Edge host : {host}   |   Label: {label}\n")
    print(f"  {'Batch':>6}   {'OK':>5}   {'Fail':>5}   "
          f"{'Total (s)':>10}   {'Avg/img (s)':>12}   "
          f"{'Data (MB)':>10}   {'MB/s':>8}")
    print("  " + "─" * (W - 2))
    for r in results:
        ok_col   = (f"\033[92m{r['successful']:>5}\033[0m" if r['failed'] == 0
                    else f"\033[93m{r['successful']:>5}\033[0m")
        fail_col = (f"\033[91m{r['failed']:>5}\033[0m" if r['failed'] > 0
                    else f"{r['failed']:>5}")
        print(f"  {r['batch_size']:>6}   {ok_col}   {fail_col}   "
              f"{r['total_time_s']:>10.3f}   {r['avg_per_img_s']:>12.4f}   "
              f"{r['total_data_mb']:>10.4f}   {r['throughput_mb_s']:>8.4f}")
    print("═" * W)
    print(f"  Results saved → {csv_path}")
    print("═" * W)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    images = load_images(args.image_dir)

    print(f"[INFO] Edge host  : {args.host}:{args.port}")
    print(f"[INFO] Remote dir : {args.remote_dir}")
    print(f"[INFO] Auth       : password")
    print(f"[INFO] Label      : {args.label}")
    print(f"[INFO] Sequence   : {BATCH_SIZES}")
    print(f"[INFO] Timeout    : {TIMEOUT_S}s")

    # Verify SSH connection once before starting
    print(f"\n[INFO] Testing SSH connection to {args.host}...", end=" ", flush=True)
    try:
        test_ssh = open_ssh(args)
        test_ssh.close()
        print("\033[92mOK\033[0m")
    except Exception as e:
        print(f"\033[91mFAILED\033[0m")
        sys.exit(f"[ERROR] Cannot connect: {e}")

    results = []
    for batch_size in BATCH_SIZES:
        r          = run_batch(args, images, batch_size)
        r["label"] = args.label
        results.append(r)

    csv_path = write_csv(results, args.label)
    print_summary(results, args.label, csv_path, args.host)


if __name__ == "__main__":
    main()