"""Chunked FlashSR inference and multi-format encoding for MulTTiPop."""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf
import torch


LOGGER = logging.getLogger(__name__)
SAMPLE_RATE = 48_000
FLASHSR_CHUNK_SAMPLES = 245_760
WEIGHT_REPO_ID = "jakeoneijk/FlashSR_weights"
WEIGHT_FILENAMES = ("student_ldm.pth", "sr_vocoder.pth", "vae.pth")
OUTPUT_FORMATS = ("opus", "mp3", "flac")
FORMAT_CODECS = {"opus": "opus", "mp3": "mp3", "flac": "flac"}


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{details}")
    return result


def _materialize_file(path: Path) -> None:
    if not path.is_symlink():
        return
    target = path.resolve(strict=True)
    temporary = path.with_name(f".{path.name}.materializing")
    temporary.write_bytes(target.read_bytes())
    path.unlink()
    os.replace(temporary, path)


def ensure_flashsr_weights(model_dir: Path) -> dict[str, Path]:
    """Download all official FlashSR checkpoints as independent local files."""
    from huggingface_hub import hf_hub_download

    model_dir = model_dir.expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    signature = inspect.signature(hf_hub_download)
    for filename in WEIGHT_FILENAMES:
        kwargs: dict[str, Any] = {
            "repo_id": WEIGHT_REPO_ID,
            "repo_type": "dataset",
            "filename": filename,
            "local_dir": str(model_dir),
        }
        if "local_dir_use_symlinks" in signature.parameters:
            kwargs["local_dir_use_symlinks"] = False
        downloaded = Path(hf_hub_download(**kwargs))
        _materialize_file(downloaded)
        if not downloaded.is_file() or downloaded.stat().st_size == 0:
            raise RuntimeError(f"missing FlashSR checkpoint: {downloaded}")
        paths[filename] = downloaded
    return paths


def chunk_starts(total_samples: int, chunk_samples: int, overlap_samples: int) -> list[int]:
    """Return full-chunk starts plus an end-aligned final chunk."""
    if total_samples <= 0:
        raise ValueError("audio must contain at least one sample")
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    if not 0 <= overlap_samples < chunk_samples:
        raise ValueError("overlap_samples must be in [0, chunk_samples)")
    if total_samples <= chunk_samples:
        return [0]
    hop = chunk_samples - overlap_samples
    final_start = total_samples - chunk_samples
    step_count = math.ceil(final_start / hop)
    # Even spacing avoids a nearly duplicate end-aligned chunk while keeping
    # every gap <= hop (and therefore every overlap >= overlap_samples).
    return [round(index * final_start / step_count) for index in range(step_count + 1)]


def overlap_window(
    starts: Sequence[int], index: int, chunk_length: int, total_samples: int
) -> torch.Tensor:
    """Build a linear cross-fade window matched to neighboring chunks."""
    start = starts[index]
    valid_length = min(chunk_length, total_samples - start)
    window = torch.ones(valid_length, dtype=torch.float32)
    if index > 0:
        previous_end = starts[index - 1] + min(
            chunk_length, total_samples - starts[index - 1]
        )
        left_overlap = max(0, previous_end - start)
        if left_overlap:
            window[:left_overlap] *= torch.linspace(0.0, 1.0, left_overlap)
    if index + 1 < len(starts):
        next_start = starts[index + 1]
        right_overlap = max(0, start + valid_length - next_start)
        if right_overlap:
            window[-right_overlap:] *= torch.linspace(1.0, 0.0, right_overlap)
    return window


def probe_audio(path: Path) -> dict[str, Any]:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,bit_rate,sample_fmt,"
            "bits_per_raw_sample:format=duration,bit_rate",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"no audio stream in {path}")
    stream = streams[0]
    format_info = payload.get("format") or {}
    duration = float(format_info.get("duration") or 0.0)
    bit_rate = int(stream.get("bit_rate") or format_info.get("bit_rate") or 0)
    return {
        "codec_name": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "duration": duration,
        "bit_rate": bit_rate,
        "sample_format": stream.get("sample_fmt"),
        "bits_per_raw_sample": int(stream.get("bits_per_raw_sample") or 0),
    }


def normalize_output_format(output_format: str) -> str:
    """Validate and normalize a supported output format name."""
    normalized = output_format.lower().lstrip(".")
    if normalized not in OUTPUT_FORMATS:
        raise ValueError(
            f"unsupported output format {output_format!r}; "
            f"choose from {', '.join(OUTPUT_FORMATS)}"
        )
    return normalized


def audio_is_valid(path: Path, output_format: str | None = None) -> bool:
    """Return whether an output is a non-empty, decodable 48 kHz audio file."""
    try:
        normalized = normalize_output_format(output_format or path.suffix)
        info = probe_audio(path)
    except Exception:
        return False
    return (
        path.is_file()
        and path.stat().st_size > 0
        and info["codec_name"] == FORMAT_CODECS[normalized]
        and info["sample_rate"] == SAMPLE_RATE
        and info["channels"] > 0
        and info["duration"] > 0
    )


def opus_is_valid(path: Path) -> bool:
    """Backward-compatible Opus-specific validation helper."""
    return audio_is_valid(path, "opus")


def build_encode_command(
    source: Path,
    destination: Path,
    output_format: str,
    opus_bitrate: str = "256k",
    mp3_bitrate: str = "320k",
    flac_compression_level: int = 8,
) -> list[str]:
    """Build an FFmpeg command for one of the supported output formats."""
    normalized = normalize_output_format(output_format)
    if not 0 <= flac_compression_level <= 12:
        raise ValueError("FLAC compression level must be between 0 and 12")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
    ]
    if normalized == "opus":
        command.extend(
            [
                "-c:a",
                "libopus",
                "-b:a",
                opus_bitrate,
                "-vbr",
                "on",
                "-compression_level",
                "10",
                "-application",
                "audio",
            ]
        )
    elif normalized == "mp3":
        command.extend(["-c:a", "libmp3lame", "-b:a", mp3_bitrate])
    else:
        command.extend(
            [
                "-c:a",
                "flac",
                "-compression_level",
                str(flac_compression_level),
                "-sample_fmt",
                "s32",
            ]
        )
    command.extend(["-ar", str(SAMPLE_RATE), str(destination)])
    return command


class FlashSRAudioProcessor:
    """Load FlashSR once and process arbitrary-length stereo audio in chunks."""

    def __init__(
        self,
        flashsr_source_dir: Path,
        model_dir: Path,
        device: str = "cuda",
        overlap_seconds: float = 0.5,
        lowpass_input: bool = True,
        opus_bitrate: str = "256k",
        output_format: str = "opus",
        mp3_bitrate: str = "320k",
        flac_compression_level: int = 8,
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        overlap_samples = round(overlap_seconds * SAMPLE_RATE)
        if not 0 <= overlap_samples < FLASHSR_CHUNK_SAMPLES:
            raise ValueError("invalid overlap duration")

        source_dir = flashsr_source_dir.expanduser().resolve()
        if not (source_dir / "FlashSR" / "FlashSR.py").is_file():
            raise FileNotFoundError(
                f"FlashSR source not found at {source_dir}; run setup_env.sh first"
            )
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        from FlashSR.FlashSR import FlashSR

        weights = ensure_flashsr_weights(model_dir)
        self.device = torch.device(device)
        self.overlap_samples = overlap_samples
        self.lowpass_input = lowpass_input
        self.opus_bitrate = opus_bitrate
        self.output_format = normalize_output_format(output_format)
        self.mp3_bitrate = mp3_bitrate
        if not 0 <= flac_compression_level <= 12:
            raise ValueError("FLAC compression level must be between 0 and 12")
        self.flac_compression_level = flac_compression_level
        LOGGER.info("Loading FlashSR on %s", self.device)
        self.model = FlashSR(
            str(weights["student_ldm.pth"]),
            str(weights["sr_vocoder.pth"]),
            str(weights["vae.pth"]),
        ).to(self.device)
        self.model.eval()

    def enhance(self, audio: torch.Tensor) -> torch.Tensor:
        """Enhance [channels, samples] float audio and return a CPU tensor."""
        if audio.ndim != 2 or audio.shape[0] < 1 or audio.shape[1] < 1:
            raise ValueError("expected non-empty [channels, samples] audio")
        audio = audio.detach().to(dtype=torch.float32, device="cpu")
        total_samples = audio.shape[-1]
        starts = chunk_starts(
            total_samples, FLASHSR_CHUNK_SAMPLES, self.overlap_samples
        )
        output = torch.zeros_like(audio)
        weights = torch.zeros(total_samples, dtype=torch.float32)

        for index, start in enumerate(starts):
            valid_length = min(FLASHSR_CHUNK_SAMPLES, total_samples - start)
            chunk = audio[:, start : start + valid_length]
            if valid_length < FLASHSR_CHUNK_SAMPLES:
                chunk = torch.nn.functional.pad(
                    chunk, (0, FLASHSR_CHUNK_SAMPLES - valid_length)
                )
            LOGGER.info(
                "FlashSR chunk %d/%d (%.2f-%.2fs)",
                index + 1,
                len(starts),
                start / SAMPLE_RATE,
                (start + valid_length) / SAMPLE_RATE,
            )
            with torch.inference_mode():
                enhanced = self.model(
                    chunk.to(self.device), lowpass_input=self.lowpass_input
                )
            enhanced = enhanced[..., :valid_length].float().cpu()
            window = overlap_window(
                starts, index, FLASHSR_CHUNK_SAMPLES, total_samples
            )
            output[:, start : start + valid_length] += enhanced * window
            weights[start : start + valid_length] += window
        output /= weights.clamp_min(torch.finfo(torch.float32).eps)
        if not torch.isfinite(output).all():
            raise RuntimeError("FlashSR produced non-finite audio")
        return output.clamp(-1.0, 1.0)

    def process(self, source: Path, destination: Path, work_root: Path) -> dict[str, Any]:
        """Decode, enhance, encode atomically, and validate one output file."""
        source = source.expanduser().resolve()
        destination = destination.expanduser().resolve()
        work_root = work_root.expanduser().resolve()
        destination_format = normalize_output_format(destination.suffix)
        if destination_format != self.output_format:
            raise ValueError(
                f"destination suffix .{destination_format} does not match "
                f"output format {self.output_format}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        work_root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="flashsr-", dir=work_root) as temporary:
            temporary_dir = Path(temporary)
            decoded = temporary_dir / "decoded.wav"
            enhanced_wav = temporary_dir / "enhanced.wav"
            encoded = temporary_dir / f"output.{self.output_format}"
            _run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-vn",
                    "-ac",
                    "2",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-c:a",
                    "pcm_f32le",
                    str(decoded),
                ]
            )
            samples, sample_rate = sf.read(
                decoded, dtype="float32", always_2d=True
            )
            if sample_rate != SAMPLE_RATE:
                raise RuntimeError(f"unexpected decoded sample rate: {sample_rate}")
            enhanced = self.enhance(torch.from_numpy(np.ascontiguousarray(samples.T)))
            sf.write(
                enhanced_wav,
                enhanced.T.numpy(),
                SAMPLE_RATE,
                subtype="FLOAT",
            )
            _run(
                build_encode_command(
                    enhanced_wav,
                    encoded,
                    self.output_format,
                    opus_bitrate=self.opus_bitrate,
                    mp3_bitrate=self.mp3_bitrate,
                    flac_compression_level=self.flac_compression_level,
                )
            )
            if not audio_is_valid(encoded, self.output_format):
                raise RuntimeError(
                    f"encoded {self.output_format.upper()} failed validation: {encoded}"
                )
            os.replace(encoded, destination)
        return probe_audio(destination)
