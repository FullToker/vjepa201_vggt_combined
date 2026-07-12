#!/usr/bin/env bash
# Downloads 3RScan (only needed if you want to use the 3rscan_* annotation
# files under leo_annotations/annotations/{alignment/scene_caption,
# alignment/obj_scene_caption}/3rscan_*.json -- those scan IDs are 3RScan
# UUIDs, not ScanNet scene0000_00-style IDs, so they need 3RScan's own scans).
#
# Wraps dataset/vendor_3rscan_download.py (official TUM CAMPAR script, saved
# verbatim). That script blocks on `input()` when downloading the "entire
# release" (no --id given) -- unusable in a non-interactive sbatch job. This
# wrapper instead fetches the scan-id list itself and loops one `--id` call
# per scan, which has no interactive prompt in the vendored script.
#
# Only pulls `sequence.zip` (RGB-D frame sequence) per scan by default --
# NOT the mesh/segs/semseg files (only needed if you later want instance
# segmentation for bbox; pass FILE_TYPE=labels.instances.annotated.v2.ply
# etc to grab those too, one type per run).
#
# By running this script you are confirming (on the user's behalf, per their
# explicit request) agreement to the 3RScan Terms of Use:
#   http://campar.in.tum.de/public_datasets/3RScan/3RScanTOU.pdf
# Read it before running if you have not already.
#
# Usage:
#   bash dataset/download_3rscan.sh [OUTPUT_DIR] [FILE_TYPE]
#   FILE_TYPE defaults to sequence.zip. Valid: mesh.refined.v2.obj,
#   mesh.refined.mtl, mesh.refined_0.png, sequence.zip,
#   labels.instances.annotated.v2.ply, mesh.refined.0.010000.segs.v2.json,
#   semseg.v2.json (the last 3 are train/val scans only, per 3RScan's release
#   notes -- the vendored script skips them automatically for test-only IDs).
#
# Dataset page: https://campar.in.tum.de/public_datasets/3RScan/3RScan.html
set -euo pipefail

BASE_URL="http://campar.in.tum.de/public_datasets/3RScan/"
OUTPUT_DIR="${1:-./source_data/3rscan}"
FILE_TYPE="${2:-sequence.zip}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUTPUT_DIR"

SCAN_LIST="$OUTPUT_DIR/.release_scans.txt"
echo "==> Fetching release scan-id list"
curl -fsSL "${BASE_URL}release_scans.txt" -o "$SCAN_LIST"

TOTAL=$(grep -cE '[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}' "$SCAN_LIST")
echo "==> ${TOTAL} scans in release_scans.txt, downloading '${FILE_TYPE}' each to: ${OUTPUT_DIR}"

i=0
grep -oE '[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}' "$SCAN_LIST" | while read -r scan_id; do
  i=$((i + 1))
  out_file="$OUTPUT_DIR/$scan_id/$FILE_TYPE"
  if [[ -f "$out_file" ]]; then
    echo "  [$i/$TOTAL] skip (exists): $scan_id"
    continue
  fi
  echo "  [$i/$TOTAL] downloading: $scan_id"
  python3 "$SCRIPT_DIR/vendor_3rscan_download.py" \
    -o "$OUTPUT_DIR" --id "$scan_id" --type "$FILE_TYPE"
done

echo "==> Done."
