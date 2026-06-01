# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FaceFusion (v3.5.2) is a face manipulation platform using ONNX Runtime for cross-platform inference. It supports face swapping, enhancement, detection, and analysis via both CLI and a Gradio web UI.

## Commands

### Installation
```bash
python install.py --onnxruntime default
# Options: default | cuda | openvino | directml | migraphx | rocm | tensorrt
# Flags: --force-reinstall, --skip-conda
```

### Running
```bash
python facefusion.py run              # Launch Gradio web UI
python facefusion.py headless-run     # Batch processing without UI
python facefusion.py batch-run        # Job-based workflow
python facefusion.py force-download   # Download ML models
python facefusion.py benchmark        # Performance testing
```

### Testing
```bash
pytest                              # Run all tests
pytest tests --cov facefusion       # With coverage
pytest tests/test_face_analyser.py  # Single test file
```

### Linting & Type Checking
```bash
flake8 facefusion.py install.py facefusion tests
mypy facefusion.py install.py facefusion tests
```

CI runs on Python 3.12 across macOS, Ubuntu, and Windows via `.github/workflows/ci.yml`.

## Architecture

**Entry point:** `facefusion.py` → `facefusion/core.py::cli()` → routes to command handlers.

### Core Pipeline

1. **Detection & Analysis** — `face_detector.py` (RetinaFace/SCRFD/YOLOFace/YUNet), `face_landmarker.py`, `face_analyser.py`, `face_recognizer.py`, `face_classifier.py`
2. **Selection & Masking** — `face_selector.py` picks faces by mode/order/attributes; `face_masker.py` builds masks; `face_store.py` caches recognized faces
3. **Processing** — Processor plugins in `facefusion/processors/modules/` apply transformations frame-by-frame
4. **Workflows** — `facefusion/workflows/image_to_image.py` and `image_to_video.py` orchestrate the full pipeline (extract frames → process → merge → restore audio → finalize)
5. **Media I/O** — `vision.py` (OpenCV frames), `ffmpeg.py` / `ffmpeg_builder.py` (video encode/decode), `audio.py` / `voice_extractor.py`

### Processor Plugin System

Each module under `facefusion/processors/modules/` implements a fixed interface:

- `get_inference_pool()` / `clear_inference_pool()` — ONNX session lifecycle
- `register_args()` / `apply_args()` — CLI argument binding
- `pre_check()` / `pre_process()` / `post_process()` — lifecycle hooks
- `process_frame(inputs)` — core per-frame transformation

Available processors: `age_modifier`, `background_remover`, `deep_swapper`, `expression_restorer`, `face_debugger`, `face_editor`, `face_enhancer`, `face_swapper`, `frame_colorizer`, `frame_enhancer`, `lip_syncer`.

Processors are loaded dynamically by `facefusion/processors/core.py`.

### State & Configuration

- `state_manager.py` — global state container with dual `cli` / `ui` contexts; use `get_item()` / `set_item()` / `sync_item()`
- `config.py` — wraps `facefusion.ini` (INI file for user defaults across all major settings)
- `choices.py` — all enums/constants (model names, encoder options, execution providers, etc.)
- `args.py` — full ArgumentParser definitions for CLI

### Job System

Jobs are persisted as JSON files in the configured `jobs_path`.

- `facefusion/jobs/job_manager.py` — CRUD operations
- `facefusion/jobs/job_runner.py` — sequential step execution
- Status flow: `drafted → queued → started → completed / failed`

### UI System

- `facefusion/uis/` — Gradio 5.44.1 web interface
- Layouts: `default.py`, `benchmark.py`, `jobs.py`, `webcam.py`
- Reusable controls in `facefusion/uis/components/`

### Inference & Execution

- `inference_manager.py` — ONNX Runtime session pooling/caching
- `execution.py` — auto-detects and prioritizes execution providers (CUDA, TensorRT, DirectML, ROCm, CoreML, OpenVINO, CPU)
- Models downloaded on demand to `.assets/models/` with hash verification via `download.py`

## Key Types (`facefusion/types.py`)

- `Face` (namedtuple): `bounding_box, score_set, landmark_set, angle, embedding, embedding_norm, gender, age, race`
- `VisionFrame`: `NDArray` — BGR uint8 image/video frame
- `ProcessMode`: `'output' | 'preview' | 'stream'`
- `ProcessState`: `'checking' | 'processing' | 'stopping' | 'pending'`

## Requirements

- **Python 3.10+** (enforced at startup in `core.py`)
- **FFmpeg** and **curl** must be on PATH
- `OMP_NUM_THREADS=1` is set for ONNX stability
- Windows Unicode paths require special handling (see `vision.py`)
