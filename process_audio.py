#!/usr/bin/env python3
"""Apply chunked FlashSR and encode one input as Opus, MP3, or FLAC."""

from __future__ import annotations

import argparse
from pathlib import Path

from audio_processing import (
    OUTPUT_FORMATS,
    FlashSRAudioProcessor,
    normalize_output_format,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=OUTPUT_FORMATS,
        help="Output format (default: infer from output filename)",
    )
    parser.add_argument("--opus-bitrate", default="256k")
    parser.add_argument("--mp3-bitrate", default="320k")
    parser.add_argument("--flac-compression-level", type=int, default=8)
    parser.add_argument("--overlap-seconds", type=float, default=0.5)
    parser.add_argument("--no-lowpass-input", action="store_true")
    parser.add_argument(
        "--flashsr-source-dir",
        type=Path,
        default=PROJECT_ROOT / "third_party" / "FlashSR_Inference",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=PROJECT_ROOT / "models" / "flashsr"
    )
    parser.add_argument(
        "--work-dir", type=Path, default=PROJECT_ROOT / ".processing-work"
    )
    args = parser.parse_args()

    try:
        output_format = normalize_output_format(
            args.output_format or args.output.suffix
        )
        suffix_format = normalize_output_format(args.output.suffix)
    except ValueError as exc:
        parser.error(str(exc))
    if suffix_format != output_format:
        parser.error(
            f"output suffix .{suffix_format} does not match --format {output_format}"
        )
    if not 0 <= args.flac_compression_level <= 12:
        parser.error("--flac-compression-level must be between 0 and 12")

    processor = FlashSRAudioProcessor(
        flashsr_source_dir=args.flashsr_source_dir,
        model_dir=args.model_dir,
        device=args.device,
        overlap_seconds=args.overlap_seconds,
        lowpass_input=not args.no_lowpass_input,
        opus_bitrate=args.opus_bitrate,
        output_format=output_format,
        mp3_bitrate=args.mp3_bitrate,
        flac_compression_level=args.flac_compression_level,
    )
    print(processor.process(args.input, args.output, args.work_dir))


if __name__ == "__main__":
    main()
