#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="${1:-${repo_root}/examples/_demo}"
python_bin="${PYTHON:-python3}"

mkdir -p "${work_dir}/dataset/example_episode"
cp "${repo_root}/examples/dataset/example_episode/metadata.json" \
  "${work_dir}/dataset/example_episode/metadata.json"

ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "testsrc2=size=640x360:rate=24:duration=6" \
  -pix_fmt yuv420p "${work_dir}/dataset/example_episode/video.mp4"

"${python_bin}" -m video_annotation_pipeline annotate-batch \
  --dataset "${work_dir}/dataset" \
  --output "${work_dir}/output" \
  --config "${repo_root}/configs/mock.yaml"

echo "Mock result: ${work_dir}/output/episodes/example_episode/review.html"
