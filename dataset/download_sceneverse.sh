#!/usr/bin/env bash
# Downloads SceneVerse language-annotation files from Google Drive directly
# onto this machine (no local-browser-then-upload round trip).
#
# SceneVerse doesn't host its own raw scans -- it publishes language
# annotations (scene-level captions etc.) keyed against scans you get
# elsewhere (ScanNet: source_data/scannet, 3RScan: source_data/3rscan,
# already present in this repo). This script only pulls SceneVerse's files,
# it does not touch the raw scan data.
#
# Needs a Google Drive file ID per file, NOT the folder. Get it from the
# Drive web UI: right-click the file -> "Get link" (or "共享" -> "复制链接"),
# the ID is the segment between /d/ and /view in the URL:
#   https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
#
# Usage:
#   bash dataset/download_sceneverse.sh <FILE_ID> <DEST_FILENAME> [OUTPUT_DIR]
#   e.g.
#   bash dataset/download_sceneverse.sh 1AbCdEfGhIjK scannet_scene_cap.json ./source_data/sceneverse
#   bash dataset/download_sceneverse.sh 1XyZ... scannet.zip ./source_data/sceneverse
#
# Requires gdown (pip install gdown). "Shared with me" files work as long as
# the owner set link-sharing to "anyone with the link" -- if gdown reports a
# permission/login error, the file is restricted to specific accounts and
# needs an authenticated flow (rclone with OAuth) instead, which this script
# does not do.
set -euo pipefail

FILE_ID="${1:?Usage: download_sceneverse.sh <FILE_ID> <DEST_FILENAME> [OUTPUT_DIR]}"
DEST_FILENAME="${2:?Usage: download_sceneverse.sh <FILE_ID> <DEST_FILENAME> [OUTPUT_DIR]}"
OUTPUT_DIR="${3:-./source_data/sceneverse}"

if ! command -v gdown >/dev/null 2>&1; then
  echo "Missing 'gdown'. Install: pip install gdown" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
DEST="$OUTPUT_DIR/$DEST_FILENAME"

if [[ -f "$DEST" ]]; then
  echo "==> Already downloaded, skipping: $DEST"
  exit 0
fi

echo "==> Downloading $FILE_ID -> $DEST"
gdown "$FILE_ID" -O "$DEST"

if [[ "$DEST_FILENAME" == *.zip ]]; then
  echo "==> Zip contents (not extracted):"
  unzip -l "$DEST" | head -50
  echo "==> To extract: unzip \"$DEST\" -d \"$OUTPUT_DIR\""
fi
