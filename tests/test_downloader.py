import json
from pathlib import Path

import pytest

from multtipop_audio import downloader
from multtipop_audio.downloader import (
    AudioRecord,
    DownloadError,
    download_youtube_segment,
    load_metadata,
)


def test_load_aggregate_metadata(tmp_path: Path) -> None:
    payload = [
        {
            "id": "sample_ID-1",
            "artist": "Artist",
            "name": "Title",
            "split": "dev",
            "youtube": {"ytid": "video_ID-1", "start": -0.5, "end": 3.25},
        }
    ]
    (tmp_path / "dev.json").write_text(json.dumps(payload))
    records = load_metadata(tmp_path)
    assert len(records) == 1
    assert records[0].dataset_id == "sample_ID-1"
    assert records[0].youtube_id == "video_ID-1"
    assert records[0].start == 0.0
    assert records[0].end == 3.25


def test_rejects_path_traversal_id(tmp_path: Path) -> None:
    payload = [
        {
            "id": "../escape",
            "youtube": {"ytid": "video_ID-1", "start": 0, "end": 1},
        }
    ]
    (tmp_path / "metadata.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid or missing dataset ID"):
        load_metadata(tmp_path)


def test_javascript_runtime_uses_project_local_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = (
        tmp_path
        / "third_party"
        / "node_runtime"
        / "node_modules"
        / "node"
        / "bin"
        / "node"
    )
    node.parent.mkdir(parents=True)
    node.write_text("")
    monkeypatch.delenv("YTDLP_NODE_PATH", raising=False)
    monkeypatch.setattr(downloader, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(downloader.shutil, "which", lambda _name: None)

    assert downloader._javascript_runtime() == {"node": {"path": str(node)}}


def test_download_retries_three_times_after_initial_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("yt_dlp")
    attempts = 0

    class FailingYoutubeDL:
        def __init__(self, options):
            assert options["retries"] == 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def download(self, _urls):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("expected failure")

    monkeypatch.setattr("yt_dlp.YoutubeDL", FailingYoutubeDL)
    monkeypatch.setattr(downloader, "_javascript_runtime", lambda: {"node": {}})
    monkeypatch.setattr(downloader.time, "sleep", lambda _seconds: None)
    record = AudioRecord("dataset_ID", "youtube_ID", 1.0, 2.0)
    with pytest.raises(DownloadError) as error:
        download_youtube_segment(record, tmp_path, retries=3)
    assert attempts == 4
    assert len(error.value.errors) == 4
