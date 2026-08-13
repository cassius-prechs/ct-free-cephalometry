#!/usr/bin/env bash
# Runs the full pipeline end to end: builds the release artifacts
# from a local data-200_pca (if not already built), then runs inference
# on a single photo.
#
# Requires:
#   - models/ populated with the trained NN checkpoints (not included in
#     this repository)
#   - data-200_pca/ (only needed for the one-time artifact-preparation
#     step; not included in this repository)
#
# Usage:
#   ./run_pipeline.sh path/to/photo.jpg [--procrustes]

set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${1:?Usage: $0 <image-path> [--procrustes]}"
shift || true

if [ ! -f models/calib_predictor.pkl ]; then
    echo "models/calib_predictor.pkl not found -- running prepare_artifacts.py..."
    python prepare_artifacts.py
fi

python infer.py --image "$IMAGE" "$@"
