import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from multtipop_audio.audio_processing import (
    SAMPLE_RATE,
    audio_is_valid,
    build_encode_command,
    chunk_starts,
    overlap_window,
    probe_audio,
)


def test_chunk_starts_end_aligns_last_chunk() -> None:
    assert chunk_starts(100, 40, 10) == [0, 30, 60]
    assert chunk_starts(101, 40, 10) == [0, 20, 41, 61]


def test_crossfade_weights_cover_every_sample() -> None:
    total = 101
    chunk = 40
    starts = chunk_starts(total, chunk, 10)
    weights = torch.zeros(total)
    for index, start in enumerate(starts):
        window = overlap_window(starts, index, chunk, total)
        weights[start : start + len(window)] += window
    assert torch.all(weights > 0)
    assert torch.allclose(weights, torch.ones_like(weights), atol=1e-6)


def test_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_starts(100, 40, 40)


@pytest.mark.parametrize(
    ("output_format", "codec"),
    [("opus", "libopus"), ("mp3", "libmp3lame"), ("flac", "flac")],
)
def test_encode_command_selects_codec(output_format: str, codec: str) -> None:
    command = build_encode_command(
        Path("input.wav"), Path(f"output.{output_format}"), output_format
    )
    assert command[command.index("-c:a") + 1] == codec
    if output_format == "mp3":
        assert command[command.index("-b:a") + 1] == "320k"


@pytest.mark.parametrize("output_format", ["opus", "mp3", "flac"])
def test_supported_format_round_trip(tmp_path: Path, output_format: str) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and FFprobe are required")
    source = tmp_path / "source.wav"
    destination = tmp_path / f"output.{output_format}"
    samples = np.zeros((SAMPLE_RATE // 10, 2), dtype=np.float32)
    sf.write(source, samples, SAMPLE_RATE, subtype="FLOAT")
    subprocess.run(
        build_encode_command(source, destination, output_format),
        check=True,
        capture_output=True,
        text=True,
    )
    assert audio_is_valid(destination, output_format)
    if output_format == "mp3":
        assert probe_audio(destination)["bit_rate"] >= 300_000
    if output_format == "flac":
        assert probe_audio(destination)["bits_per_raw_sample"] == 24


def test_rejects_unknown_output_format() -> None:
    with pytest.raises(ValueError, match="unsupported output format"):
        build_encode_command(Path("input.wav"), Path("output.aac"), "aac")
