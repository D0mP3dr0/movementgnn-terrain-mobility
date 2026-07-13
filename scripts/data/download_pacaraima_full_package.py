"""
Full-package download orchestrator for the Pacaraima study area:
- DEM (Copernicus GLO-30)
- Sentinel-2 (Q1-Q4, 10m)
- LiDAR (ICESat-2 + GEDI)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent


def run_cmd(cmd: list[str]) -> int:
    print(f"\n>>> {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=str(ROOT.parent.parent))
    return p.returncode


def main():
    parser = argparse.ArgumentParser(description="Full download package for Pacaraima.")
    parser.add_argument("--skip-dem", action="store_true")
    parser.add_argument("--skip-sentinel", action="store_true")
    parser.add_argument("--skip-lidar", action="store_true")
    parser.add_argument("--max-cloud", type=float, default=20.0)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--year-start", type=int, default=2023)
    parser.add_argument("--max-scenes", type=int, default=20)
    parser.add_argument("--dry-run-sentinel", action="store_true")
    parser.add_argument("--allow-netrc-lidar", action="store_true")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_dem:
        rc = run_cmd([py, str(ROOT / "download_dem_pacaraima_q1q4.py")])
        if rc != 0:
            sys.exit(rc)

    if not args.skip_sentinel:
        cmd = [
            py, str(ROOT / "download_sentinel2_pacaraima_q1q4.py"),
            "--max-cloud", str(args.max_cloud),
            "--year", str(args.year),
            "--year-start", str(args.year_start),
            "--max-scenes", str(args.max_scenes),
        ]
        if args.dry_run_sentinel:
            cmd.append("--dry-run")
        rc = run_cmd(cmd)
        if rc != 0:
            sys.exit(rc)

    if not args.skip_lidar:
        cmd = [py, str(ROOT / "download_lidar_pacaraima_q1q4.py")]
        if args.allow_netrc_lidar:
            cmd.append("--allow-netrc")
        rc = run_cmd(cmd)
        if rc != 0:
            sys.exit(rc)

    print("\nFull package download completed.")


if __name__ == "__main__":
    main()
