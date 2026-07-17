#!/usr/bin/env python3
"""Run the complete MulTTiPop download, FlashSR, and audio encoding pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from audio_processing import OUTPUT_FORMATS, FlashSRAudioProcessor, audio_is_valid
from downloader import (
    DEFAULT_REPO_ID,
    AudioRecord,
    DownloadError,
    download_dataset,
    download_youtube_segment,
    load_metadata,
)


LOGGER = logging.getLogger("multtipop_pipeline")
PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=os.environ.get("DATA_ROOT"),
        help="Dataset root (default: $DATA_ROOT)",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--cookies", type=Path)
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=OUTPUT_FORMATS,
        default="opus",
        help="Output format (default: opus)",
    )
    parser.add_argument("--opus-bitrate", default="256k")
    parser.add_argument("--mp3-bitrate", default="320k")
    parser.add_argument("--flac-compression-level", type=int, default=8)
    parser.add_argument("--overlap-seconds", type=float, default=0.5)
    parser.add_argument(
        "--no-lowpass-input",
        action="store_true",
        help="Disable FlashSR's automatic input low-pass filtering",
    )
    parser.add_argument("--skip-dataset-download", action="store_true")
    parser.add_argument("--keep-source", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--id", dest="ids", action="append", help="Only process this ID; repeatable"
    )
    parser.add_argument(
        "--flashsr-source-dir",
        type=Path,
        default=PROJECT_ROOT / "third_party" / "FlashSR_Inference",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=PROJECT_ROOT / "models" / "flashsr"
    )
    args = parser.parse_args(argv)
    if args.data_root is None:
        parser.error("--data-root is required when $DATA_ROOT is not set")
    if args.retries < 0:
        parser.error("--retries must be non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not 0 <= args.flac_compression_level <= 12:
        parser.error("--flac-compression-level must be between 0 and 12")
    return args


def configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)
    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    for noisy_logger in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_failed_ids(path: Path, failed: dict[str, str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("".join(f"{item_id}\t{stage}\n" for item_id, stage in failed.items()))
    os.replace(temporary, path)


def choose_records(
    records: Sequence[AudioRecord], ids: Sequence[str] | None, limit: int | None
) -> list[AudioRecord]:
    selected = list(records)
    if ids:
        wanted = set(ids)
        selected = [record for record in selected if record.dataset_id in wanted]
        missing = wanted - {record.dataset_id for record in selected}
        if missing:
            raise ValueError(f"unknown requested IDs: {', '.join(sorted(missing))}")
    if limit is not None:
        selected = selected[:limit]
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    dataset_dir = data_root / "multtipop"
    audio_dir = dataset_dir / "audio"
    work_dir = dataset_dir / ".work"
    log_dir = dataset_dir / "logs"
    configure_logging(log_dir)

    if not args.skip_dataset_download:
        download_dataset(dataset_dir, args.repo_id, args.revision)
    records = choose_records(load_metadata(dataset_dir), args.ids, args.limit)
    audio_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "downloads").mkdir(parents=True, exist_ok=True)
    run_started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    state: dict[str, Any] = {
        "state": "running",
        "started_at": run_started,
        "updated_at": run_started,
        "dataset_dir": str(dataset_dir),
        "output_format": args.output_format,
        "total": len(records),
        "processed": 0,
        "succeeded": 0,
        "skipped": 0,
        "failed": 0,
        "current_id": None,
    }
    status_path = log_dir / "status.json"
    failed_ids_path = log_dir / "failed_ids.log"
    failures_path = log_dir / "failures.jsonl"
    write_json_atomic(status_path, state)
    write_failed_ids(failed_ids_path, {})

    processor: FlashSRAudioProcessor | None = None
    failed: dict[str, str] = {}
    try:
        for index, record in enumerate(records, start=1):
            state["current_id"] = record.dataset_id
            state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            write_json_atomic(status_path, state)
            destination = audio_dir / f"{record.dataset_id}.{args.output_format}"
            LOGGER.info("[%d/%d] %s - %s", index, len(records), record.dataset_id, record.title)
            if not args.overwrite and audio_is_valid(destination, args.output_format):
                LOGGER.info("Skipping valid output %s", destination)
                state["skipped"] += 1
                state["processed"] += 1
                write_json_atomic(status_path, state)
                continue

            try:
                source = download_youtube_segment(
                    record,
                    work_dir / "downloads",
                    retries=args.retries,
                    cookies=args.cookies,
                )
            except DownloadError as exc:
                failure = {
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "stage": "download",
                    "record": record.to_dict(),
                    "attempts": len(exc.errors),
                    "errors": exc.errors,
                }
                append_jsonl(failures_path, failure)
                append_jsonl(log_dir / "download_failures.jsonl", failure)
                failed[record.dataset_id] = "download"
                state["failed"] += 1
                state["processed"] += 1
                write_failed_ids(failed_ids_path, failed)
                write_json_atomic(status_path, state)
                continue

            try:
                if processor is None:
                    processor = FlashSRAudioProcessor(
                        flashsr_source_dir=args.flashsr_source_dir,
                        model_dir=args.model_dir,
                        device=args.device,
                        overlap_seconds=args.overlap_seconds,
                        lowpass_input=not args.no_lowpass_input,
                        opus_bitrate=args.opus_bitrate,
                        output_format=args.output_format,
                        mp3_bitrate=args.mp3_bitrate,
                        flac_compression_level=args.flac_compression_level,
                    )
                info = processor.process(source, destination, work_dir / "processing")
                LOGGER.info("Created %s (%s)", destination, info)
                state["succeeded"] += 1
                if not args.keep_source:
                    shutil.rmtree(source.parent, ignore_errors=True)
            except Exception as exc:
                LOGGER.exception("Processing failed for %s", record.dataset_id)
                failure = {
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "stage": "processing",
                    "record": record.to_dict(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "source": str(source),
                    "output_format": args.output_format,
                }
                append_jsonl(failures_path, failure)
                append_jsonl(log_dir / "processing_failures.jsonl", failure)
                failed[record.dataset_id] = "processing"
                state["failed"] += 1
            state["processed"] += 1
            write_failed_ids(failed_ids_path, failed)
            write_json_atomic(status_path, state)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; rerun the same command to resume")
        state["state"] = "interrupted"
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json_atomic(status_path, state)
        return 130

    state["state"] = "completed_with_failures" if failed else "completed"
    state["current_id"] = None
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json_atomic(status_path, state)
    LOGGER.info(
        "Finished: succeeded=%d skipped=%d failed=%d total=%d",
        state["succeeded"],
        state["skipped"],
        state["failed"],
        state["total"],
    )
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
