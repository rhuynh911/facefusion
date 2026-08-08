#!/usr/bin/env bash
#
# Launch FaceFusion on the NVIDIA GPU from the local venv.
#
# Why this exists: the system Python is 3.14, which has no wheels for the pinned
# dependency set, so everything lives in ./.venv on Python 3.12. Any extra
# arguments are passed straight through, e.g.
#
#   ./run-gpu.sh headless-run -s face.jpg -t clip.mp4 -o out.mp4
#
# CUDA and cuDNN come from the nvidia site packages and are loaded by
# facefusion/execution.py, so no LD_LIBRARY_PATH is needed. Execution provider
# (cuda) and video encoder (h264_nvenc, since Fedora's ffmpeg has no libx264)
# are set in facefusion.ini.

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

if [ ! -x .venv/bin/python ]; then
	echo "run-gpu.sh: .venv is missing — create it with:" >&2
	echo "  uv python install 3.12 && \$(uv python find 3.12) -m venv .venv" >&2
	echo "  .venv/bin/pip install -r requirements.txt" >&2
	echo "  .venv/bin/pip install 'onnxruntime-gpu[cuda,cudnn]==1.24.4'" >&2
	exit 1
fi

if ! .venv/bin/python -c 'import onnxruntime, sys; sys.exit(0 if "CUDAExecutionProvider" in onnxruntime.get_available_providers() else 1)' 2>/dev/null; then
	echo "run-gpu.sh: onnxruntime has no CUDA provider — install it with:" >&2
	echo "  .venv/bin/pip uninstall -y onnxruntime" >&2
	echo "  .venv/bin/pip install 'onnxruntime-gpu[cuda,cudnn]==1.24.4'" >&2
	exit 1
fi

exec .venv/bin/python facefusion.py "${@:-run}"
