"""Dataset and YouTube segment download support for MulTTiPop."""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


LOGGER = logging.getLogger(__name__)
DEFAULT_REPO_ID = "gclef-cmu/multtipop"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class DownloadError(RuntimeError):
    """Raised after all YouTube download attempts fail."""

    def __init__(self, dataset_id: str, errors: Sequence[str]):
        self.dataset_id = dataset_id
        self.errors = list(errors)
        super().__init__(
            f"download failed for {dataset_id} after {len(errors)} attempt(s): "
            f"{errors[-1] if errors else 'unknown error'}"
        )


@dataclass(frozen=True)
class AudioRecord:
    """One requested MulTTiPop audio segment."""

    dataset_id: str
    youtube_id: str
    start: float
    end: float
    split: str = ""
    artist: str = ""
    title: str = ""

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.youtube_id}"

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["url"] = self.url
        data["duration"] = self.duration
        return data


def _materialize_symlink(path: Path) -> None:
    target = path.resolve(strict=True)
    temporary = path.with_name(f".{path.name}.materializing")
    if target.is_dir():
        shutil.copytree(target, temporary)
        path.unlink()
        temporary.replace(path)
        return
    shutil.copy2(target, temporary)
    path.unlink()
    temporary.replace(path)


def materialize_all_symlinks(root: Path) -> int:
    """Replace every symlink below root with an independent file/directory copy."""
    count = 0
    # Deepest paths first so a directory symlink cannot hide child links.
    links = sorted(
        (path for path in root.rglob("*") if path.is_symlink()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in links:
        _materialize_symlink(path)
        count += 1
    return count


def assert_no_symlinks(root: Path) -> None:
    links = [path for path in root.rglob("*") if path.is_symlink()]
    if links:
        preview = ", ".join(str(path) for path in links[:5])
        raise RuntimeError(f"dataset still contains symlinks: {preview}")


def download_dataset(
    dataset_dir: Path,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = "main",
) -> Path:
    """Download the complete Hugging Face dataset using real local files."""
    from huggingface_hub import HfApi, snapshot_download

    dataset_dir = dataset_dir.expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": revision,
        "local_dir": str(dataset_dir),
    }
    # Older huggingface_hub releases need this explicit opt-out.
    if "local_dir_use_symlinks" in inspect.signature(snapshot_download).parameters:
        kwargs["local_dir_use_symlinks"] = False

    LOGGER.info("Downloading dataset %s@%s to %s", repo_id, revision, dataset_dir)
    snapshot_download(**kwargs)
    materialized = materialize_all_symlinks(dataset_dir)
    assert_no_symlinks(dataset_dir)

    info = HfApi().dataset_info(repo_id=repo_id, revision=revision)
    manifest = {
        "repo_id": repo_id,
        "requested_revision": revision,
        "commit_sha": info.sha,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "materialized_symlink_count": materialized,
    }
    _write_json_atomic(dataset_dir / ".dataset_download.json", manifest)
    LOGGER.info("Dataset ready (commit %s, symlinks remaining: 0)", info.sha)
    return dataset_dir


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _records_from_payload(payload: Any, source: Path) -> list[AudioRecord]:
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            rows: Iterable[Any] = payload["data"]
        elif isinstance(payload.get("records"), list):
            rows = payload["records"]
        else:
            rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"unsupported JSON payload in {source}")

    records: list[AudioRecord] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        records.append(_parse_record(row, source))
    return records


def _youtube_id(row: dict[str, Any], youtube: Any) -> str:
    if isinstance(youtube, dict):
        for key in ("ytid", "id", "video_id", "videoId"):
            if youtube.get(key):
                return str(youtube[key])
    elif isinstance(youtube, str):
        match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})", youtube)
        if match:
            return match.group(1)

    for key in ("youtube_id", "youtubeId", "ytid", "audio_link", "url"):
        value = row.get(key)
        if not value:
            continue
        match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})", str(value))
        if match:
            return match.group(1)
        if key in {"youtube_id", "youtubeId", "ytid"}:
            return str(value)
    raise ValueError("missing YouTube video ID")


def _parse_record(row: dict[str, Any], source: Path) -> AudioRecord:
    dataset_id = str(row.get("id") or row.get("ID") or "")
    if not _SAFE_ID.fullmatch(dataset_id):
        raise ValueError(f"invalid or missing dataset ID {dataset_id!r} in {source}")

    youtube = row.get("youtube") or {}
    youtube_id = _youtube_id(row, youtube)
    if not _SAFE_ID.fullmatch(youtube_id):
        raise ValueError(f"invalid YouTube ID {youtube_id!r} for {dataset_id}")

    if isinstance(youtube, dict):
        start_value = youtube.get("start", row.get("start", 0))
        end_value = youtube.get("end", row.get("end"))
    else:
        start_value = row.get("start", 0)
        end_value = row.get("end")
    start = max(0.0, float(start_value or 0.0))
    if end_value is None:
        raise ValueError(f"missing end timestamp for {dataset_id}")
    end = float(end_value)
    if end <= start:
        raise ValueError(f"invalid time range [{start}, {end}] for {dataset_id}")

    return AudioRecord(
        dataset_id=dataset_id,
        youtube_id=youtube_id,
        start=start,
        end=end,
        split=str(row.get("split") or row.get("split_name") or ""),
        artist=str(row.get("artist") or ""),
        title=str(row.get("name") or row.get("title") or ""),
    )


def load_metadata(dataset_dir: Path) -> list[AudioRecord]:
    """Load aggregate metadata, with a per-item meta.json fallback."""
    dataset_dir = dataset_dir.expanduser().resolve()
    aggregate_files = [
        path
        for path in (
            dataset_dir / "metadata.json",
            dataset_dir / "dev.json",
            dataset_dir / "test.json",
        )
        if path.is_file()
    ]
    if aggregate_files:
        sources = aggregate_files
    else:
        sources = sorted(
            set(dataset_dir.rglob("meta.json"))
            | set(dataset_dir.rglob("metadata.json"))
        )
    if not sources:
        raise FileNotFoundError(f"no MulTTiPop metadata JSON found under {dataset_dir}")

    by_id: dict[str, AudioRecord] = {}
    for source in sources:
        payload = json.loads(source.read_text())
        for record in _records_from_payload(payload, source):
            existing = by_id.get(record.dataset_id)
            if existing and existing != record:
                raise ValueError(f"conflicting metadata for ID {record.dataset_id}")
            by_id[record.dataset_id] = record
    records = sorted(by_id.values(), key=lambda item: (item.split, item.dataset_id))
    if not records:
        raise ValueError(f"metadata under {dataset_dir} contained no records")
    LOGGER.info("Loaded %d unique metadata records", len(records))
    return records


def _media_is_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0 or ".part" in path.name:
        return False
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _clean_download_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _javascript_runtime() -> dict[str, dict[str, str]]:
    override = os.environ.get("YTDLP_NODE_PATH")
    local_node = (
        Path(__file__).resolve().parent
        / "third_party"
        / "node_runtime"
        / "node_modules"
        / "node"
        / "bin"
        / "node"
    )
    path = override or (str(local_node) if local_node.is_file() else shutil.which("node"))
    if not path:
        raise RuntimeError(
            "no JavaScript runtime found; run setup_env.sh or set YTDLP_NODE_PATH"
        )
    return {"node": {"path": path}}


def download_youtube_segment(
    record: AudioRecord,
    staging_root: Path,
    retries: int = 3,
    cookies: Path | None = None,
) -> Path:
    """Download the requested segment, retrying `retries` times after the first try."""
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import download_range_func

    if retries < 0:
        raise ValueError("retries must be non-negative")
    item_dir = staging_root.expanduser().resolve() / record.dataset_id
    item_dir.mkdir(parents=True, exist_ok=True)
    for existing in sorted(item_dir.glob("source.*")):
        if _media_is_valid(existing):
            LOGGER.info("Reusing staged source for %s: %s", record.dataset_id, existing)
            return existing

    errors: list[str] = []
    max_attempts = retries + 1
    for attempt in range(1, max_attempts + 1):
        _clean_download_directory(item_dir)
        output_template = str(item_dir / "source.%(ext)s")
        options: dict[str, Any] = {
            "format": "bestaudio/best",
            # Prefer sample rate first, then bitrate among audio-only formats.
            "format_sort": ["asr", "abr"],
            "outtmpl": output_template,
            "download_ranges": download_range_func(
                None, [(record.start, record.end)]
            ),
            "force_keyframes_at_cuts": True,
            "noplaylist": True,
            "retries": 0,
            "fragment_retries": 0,
            "extractor_retries": 0,
            "file_access_retries": 0,
            "socket_timeout": 30,
            "quiet": True,
            "no_warnings": False,
            "js_runtimes": _javascript_runtime(),
        }
        if cookies is not None:
            options["cookiefile"] = str(cookies.expanduser().resolve())

        LOGGER.info(
            "Downloading %s (%s, %.3f-%.3fs), attempt %d/%d",
            record.dataset_id,
            record.youtube_id,
            record.start,
            record.end,
            attempt,
            max_attempts,
        )
        try:
            with YoutubeDL(options) as downloader:
                downloader.download([record.url])
            candidates = [
                path
                for path in sorted(item_dir.glob("source.*"))
                if _media_is_valid(path)
            ]
            if not candidates:
                raise RuntimeError("yt-dlp returned success but produced no valid audio")
            return candidates[0]
        except Exception as exc:  # yt-dlp exposes multiple exception subclasses.
            message = f"{type(exc).__name__}: {exc}"
            errors.append(message)
            LOGGER.warning("Attempt %d failed for %s: %s", attempt, record.dataset_id, message)
            if attempt < max_attempts:
                time.sleep(min(30.0, 2.0 ** (attempt - 1)))
    raise DownloadError(record.dataset_id, errors)
