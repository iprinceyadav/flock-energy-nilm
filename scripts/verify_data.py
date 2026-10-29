#!/usr/bin/env python3
"""
verify_data.py — Verify dataset availability and structure.
Run this first to confirm data files are present and readable.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DIR, CLEANED_DIR, WM_CHANNEL_MAP


def main():
    print("=" * 60)
    print("REFIT Dataset Verification")
    print("=" * 60)

    # Check raw files
    print("\n-- Raw Data --")
    raw_files = sorted(RAW_DIR.glob("*.csv")) if RAW_DIR.exists() else []
    print(f"Directory: {RAW_DIR}")
    print(f"Files found: {len(raw_files)}")
    for f in raw_files:
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

    # Check House 1 raw
    h1_raw = list(RAW_DIR.glob("RAW_House1_Part*.csv"))
    print(f"\nHouse 1 raw files: {len(h1_raw)} {'[OK]' if len(h1_raw) == 2 else '[MISSING]'}")

    # Check cleaned files
    print("\n-- Cleaned Data --")
    clean_files = sorted(CLEANED_DIR.glob("*.csv")) if CLEANED_DIR.exists() else []
    print(f"Directory: {CLEANED_DIR}")
    print(f"Files found: {len(clean_files)}")
    for f in clean_files:
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

    # Check WM channel availability
    print("\n-- Washing Machine Channels --")
    for house_id, channel in sorted(WM_CHANNEL_MAP.items()):
        clean_file = CLEANED_DIR / f"CLEAN_House{house_id}.csv"
        status = "FOUND" if clean_file.exists() else "MISSING"
        print(f"  House {house_id:2d}: {channel:12s} [{status}]")

    print("\n" + "=" * 60)
    print("Verification complete")


if __name__ == "__main__":
    main()
