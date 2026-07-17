# MulTTiPop Audio Pipeline

This project downloads the complete
[`gclef-cmu/multtipop`](https://huggingface.co/datasets/gclef-cmu/multtipop)
dataset, obtains only the YouTube time ranges referenced by its metadata, applies
[FlashSR](https://github.com/jakeoneijk/FlashSR_Inference), and writes 48 kHz
audio files to:

```text
$DATA_ROOT/multtipop/audio/{ID}.{opus|mp3|flac}
```

Opus is the default. MP3 is encoded at 320 kbit/s CBR by default, while FLAC is
lossless. The pipeline is resumable: existing valid outputs for the selected
format are skipped, successfully downloaded source segments are reused after
processing failures, final outputs are written atomically, and each download
receives one initial attempt plus three retries by default.

## What the pipeline does

1. Downloads the Hugging Face dataset snapshot to `$DATA_ROOT/multtipop`.
2. Replaces and verifies all symlinks so the dataset consists of real local files.
3. Reads `metadata.json`, `dev.json`/`test.json`, or per-item `meta.json` files.
4. Uses `yt-dlp` to select audio by sample rate and then bitrate, and downloads only
   each metadata time range.
5. Decodes to 48 kHz stereo and runs the official FlashSR model in overlapping
   5.12-second chunks. Linear cross-fades remove chunk boundary discontinuities.
6. Encodes the result as Opus (`libopus`, 256 kbit/s VBR), MP3 (`libmp3lame`,
   320 kbit/s CBR), or lossless 24-bit FLAC. The codec, sample rate, channel
   count, and duration are validated with `ffprobe` before publishing the final
   file.

## Requirements

- Linux with an NVIDIA GPU and a CUDA-compatible driver
- Conda (the setup script defaults to `/home/work/miniforge/bin/conda`)
- Git, FFmpeg/FFprobe with `libopus`, `libmp3lame`, and FLAC support, npm, and
  Node.js
- Enough space for the dataset, temporary media, and approximately 3.4 GB of
  FlashSR checkpoints

## Installation

```bash
cd /home/work/G2SMusic/sungkyun/amt/multtipop-audio
bash setup_env.sh
```

The script creates `/home/work/miniforge/envs/multtipop_audio`, installs a CUDA
12.6 PyTorch build, checks out a pinned FlashSR revision, installs the Python
dependencies, and installs a project-local Node.js 22 runtime for yt-dlp.
Override `CONDA_BIN`, `ENV_NAME`, or `ENV_PREFIX` when needed.

Select a data root. `run_pipeline.sh` invokes the environment directly and
isolates it from user-level Python packages:

```bash
export DATA_ROOT=/path/to/data
```

## Run everything

```bash
./run_pipeline.sh --data-root "$DATA_ROOT"
```

This produces `$DATA_ROOT/multtipop/audio/{ID}.opus`. Select another format
without changing the rest of the workflow:

```bash
# 320 kbit/s MP3
./run_pipeline.sh --data-root "$DATA_ROOT" --format mp3

# Lossless FLAC
./run_pipeline.sh --data-root "$DATA_ROOT" --format flac
```

Outputs of different formats can coexist because they use different filename
extensions. Resume and overwrite checks apply only to the selected format.

On the current `kt` server, the established AMT data root is:

```bash
./run_pipeline.sh --data-root /home/work/sungkyun/amt/data
```

Useful options:

```text
--retries 3              Retries after the initial yt-dlp attempt
--cookies FILE           Netscape-format YouTube cookie file
--format opus|mp3|flac   Select output format (default: opus)
--id ID                  Process one ID (repeat to select several)
--limit N                Process the first N selected records
--keep-source            Retain downloaded source segments after success
--overwrite              Regenerate valid files in the selected format
--skip-dataset-download  Use an already downloaded dataset snapshot
--device cuda:0          Select the inference device
--opus-bitrate 256k      Set the Opus bitrate
--mp3-bitrate 320k       Set the MP3 constant bitrate
--flac-compression-level 8
                         Set FLAC compression level (0-12; lossless at all levels)
--no-lowpass-input       Disable FlashSR's automatic low-pass preprocessing
```

To run persistently over SSH:

```bash
mkdir -p "$DATA_ROOT/multtipop/logs"
nohup ./run_pipeline.sh --data-root "$DATA_ROOT" \
  > "$DATA_ROOT/multtipop/logs/launcher.log" 2>&1 &
echo $! > "$DATA_ROOT/multtipop/logs/pipeline.pid"
```

Rerun the same command, including the same `--format`, after interruption.
Completed files are detected and skipped automatically. Run only one complete
pipeline at a time because all formats share the same progress and failure log
files.

## Monitor and troubleshoot

```bash
cat "$DATA_ROOT/multtipop/logs/status.json"
tail -f "$DATA_ROOT/multtipop/logs/launcher.log"
cat "$DATA_ROOT/multtipop/logs/failed_ids.log"
```

Log files:

- `status.json`: atomic progress snapshot, including counts and the current ID
- `pipeline.log`: detailed application log
- `launcher.log`: stdout/stderr from the background process
- `failed_ids.log`: current failed IDs and failure stage
- `download_failures.jsonl`: records whose four download attempts all failed
- `processing_failures.jsonl`: FlashSR or output encoding failures
- `failures.jsonl`: combined machine-readable failure history

YouTube periodically requires authentication or changes its player. Update
`yt-dlp` first if extraction starts failing. If YouTube reports that sign-in is
required, export browser cookies in Netscape format and pass `--cookies FILE`.
Do not commit cookie files.

## Individual stages

Download source segments only (the dataset snapshot must already exist):

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 \
  /home/work/miniforge/envs/multtipop_audio/bin/python \
  download_audio.py --data-root "$DATA_ROOT" --id ANmpGEL-oyM
```

Apply FlashSR and encode one local source file. The format is inferred from the
output filename:

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 \
  /home/work/miniforge/envs/multtipop_audio/bin/python \
  process_audio.py input.webm output.opus

PYTHONPATH= PYTHONNOUSERSITE=1 \
  /home/work/miniforge/envs/multtipop_audio/bin/python \
  process_audio.py input.webm output.mp3

PYTHONPATH= PYTHONNOUSERSITE=1 \
  /home/work/miniforge/envs/multtipop_audio/bin/python \
  process_audio.py input.webm output.flac
```

`process_audio.py` also accepts `--format`, `--opus-bitrate`, `--mp3-bitrate`,
and `--flac-compression-level`. When `--format` is supplied, it must match the
output filename extension.

The complete workflow should normally use `run_pipeline.py`, because it provides
resume checks, failure logs, cleanup, progress reporting, and atomic output.

## Tests

```bash
pytest -q
```

## Disclaimer

This unofficial repository is not created or endorsed by the MulTTiPop authors
or YouTube. All rights in the referenced music and recordings remain with their
respective copyright holders; this repository grants no rights to that content.
It is intended only for non-commercial research. Users are solely responsible
for legal and license compliance. The software is provided as-is and used at
your own risk; the maintainers accept no liability.

## Citation

If you use [MulTTiPop](https://huggingface.co/datasets/gclef-cmu/multtipop),
please cite the [original paper](https://arxiv.org/abs/2607.08756):

```bibtex
@article{pruyne2026multtipop,
  title={MulTTiPop: A Multitrack Transcription Dataset for Pop Music},
  author={Pruyne, Nathan and Stoler, Benjamin and Chen, William and Huang,
          Chien-yu and Watanabe, Shinji and Donahue, Chris},
  journal={arXiv preprint arXiv:2607.08756},
  year={2026}
}
```
