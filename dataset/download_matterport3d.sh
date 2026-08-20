#!/usr/bin/env bash
# Downloads Matterport3D (only needed for MMScan's own VG/QA/Caption benchmark --
# every scan_id under MMScan-beta/MMScan_samples/*.json is matterport3d/<house>/regionN,
# and none of that is usable without the raw posed images those scan_ids point at).
#
# Wraps dataset/vendor_matterport_download.py (TUM-official download_mp.py,
# ported to py3 + non-interactive -- see that file's header. The unmodified
# original python2 script is kept verbatim at ckpts/download_mp.py).
#
# Only pulls undistorted_color_images + undistorted_camera_parameters (posed
# RGB, what preprocess()/manifest-building needs) + region_segmentations
# (region-level annotation, matches MMScan's region_id) by default -- NOT the
# full 17 file types (--type ALL below would need ~1.3TB). Override FILE_TYPES
# to grab more.
#
# By running this script you are confirming (on the user's explicit request)
# agreement to the Matterport3D Terms of Use:
#   http://kaldir.vc.cit.tum.de/matterport/MP_TOS.pdf
# Read it before running if you have not already.
#
# Usage:
#   bash dataset/download_matterport3d.sh [OUTPUT_DIR] [FILE_TYPES...]
#   e.g. bash dataset/download_matterport3d.sh ./source_data/matterport3d undistorted_color_images
set -euo pipefail

OUTPUT_DIR="${1:-./source_data/matterport3d}"
shift || true
FILE_TYPES=("$@")
if [ "${#FILE_TYPES[@]}" -eq 0 ]; then
    FILE_TYPES=(undistorted_color_images undistorted_camera_parameters region_segmentations)
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUTPUT_DIR"

echo "==> Downloading Matterport3D file types [${FILE_TYPES[*]}] to: $OUTPUT_DIR"
echo "==> This covers all ~90 houses in one run (vendor_matterport_download.py skips already-downloaded files, so re-running resumes)."

python3 "$SCRIPT_DIR/vendor_matterport_download.py" \
    -o "$OUTPUT_DIR" --id ALL --type "${FILE_TYPES[@]}" --yes

echo "==> Done."
