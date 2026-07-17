#!/usr/bin/env python3
"""Download MulTTiPop YouTube segments without running FlashSR."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from downloader import download_youtube_segment, load_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("DATA_ROOT"))
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--cookies", type=Path)
    parser.add_argument("--id", dest="ids", action="append")
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("--data-root is required when $DATA_ROOT is not set")
    dataset_dir = args.data_root.expanduser().resolve() / "multtipop"
    records = load_metadata(dataset_dir)
    if args.ids:
        wanted = set(args.ids)
        records = [record for record in records if record.dataset_id in wanted]
    staging = dataset_dir / ".work" / "downloads"
    for index, record in enumerate(records, 1):
        path = download_youtube_segment(record, staging, args.retries, args.cookies)
        print(f"[{index}/{len(records)}] {record.dataset_id}\t{path}")


if __name__ == "__main__":
    main()
