#!/usr/bin/env bash
# Downloads ScanNet's official per-frame 2D instance masks (_2d-instance-filt.zip)
# for every train/val scan, used to resolve target_id -> instance mask/bbox
# (see dataset/download_leo_annotations.sh's note on resolving target_id
# against ScanNet instance annotations).
#
# Wraps dataset/vendor_scannet_download.py (official kaldir script, saved
# verbatim). That script blocks on input() unconditionally at the top of
# main() (TOS confirmation) even for a single --id download -- this wrapper
# pipes it a blank line so it's non-interactive in an sbatch job.
#
# _2d-instance-filt.zip is a train/val-only file type (not part of the
# hidden test release), so only v2/scans.txt scenes are used here.
#
# Resumable: a scan whose instance-filt/ dir already exists is skipped, so
# re-running after a partial/interrupted run only fills gaps. Existing
# posed_images/ or other scene data is untouched -- output goes to
# <OUTPUT_DIR>/scans/<scene_id>/instance-filt/.
#
# By running this script you are confirming (on the user's behalf, per their
# explicit request) agreement to the ScanNet Terms of Use:
#   https://kaldir.vc.cit.tum.de/scannet/ScanNet_TOS.pdf
#
# Usage:
#   bash dataset/download_scannet_2d_instance.sh [OUTPUT_DIR] [LIMIT]
#   OUTPUT_DIR defaults to ./source_data/scannet (same root as posed_images/).
#   LIMIT, if set, only processes the first N scenes (smoke test).
set -euo pipefail

BASE_URL="http://kaldir.vc.cit.tum.de/scannet/"
OUTPUT_DIR="${1:-./source_data/scannet}"
LIMIT="${2:-0}"
FILE_TYPE="_2d-instance-filt.zip"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUTPUT_DIR"

SCAN_LIST="$OUTPUT_DIR/.v2_scans.txt"
echo "==> Fetching v2 train/val scan-id list"
curl -fsSL "${BASE_URL}v2/scans.txt" -o "$SCAN_LIST"

TOTAL=$(wc -l < "$SCAN_LIST")
if [[ "$LIMIT" != "0" ]]; then
  head -n "$LIMIT" "$SCAN_LIST" > "$SCAN_LIST.limited"
  mv "$SCAN_LIST.limited" "$SCAN_LIST"
  TOTAL="$LIMIT"
fi
echo "==> ${TOTAL} scans, downloading '${FILE_TYPE}' each to: ${OUTPUT_DIR}/scans/<id>/instance-filt/"

i=0
while read -r scan_id; do
  [[ -z "$scan_id" ]] && continue
  i=$((i + 1))
  scan_dir="$OUTPUT_DIR/scans/$scan_id"
  zip_file="$scan_dir/${scan_id}${FILE_TYPE}"
  extracted_dir="$scan_dir/instance-filt"

  if [[ -d "$extracted_dir" ]]; then
    echo "  [$i/$TOTAL] skip (already extracted): $scan_id"
    continue
  fi

  echo "  [$i/$TOTAL] downloading: $scan_id"
  echo | python3 "$SCRIPT_DIR/vendor_scannet_download.py" \
    -o "$OUTPUT_DIR" --id "$scan_id" --type "$FILE_TYPE" --skip_existing

  if [[ -f "$zip_file" ]]; then
    echo "  [$i/$TOTAL] extracting: $scan_id"
    mkdir -p "$extracted_dir"
    python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
      "$zip_file" "$scan_dir"
    rm -f "$zip_file"
  else
    echo "  [$i/$TOTAL] WARNING: no zip produced for $scan_id (bad scan id or download error)"
  fi
done < "$SCAN_LIST"

echo "==> Done."
