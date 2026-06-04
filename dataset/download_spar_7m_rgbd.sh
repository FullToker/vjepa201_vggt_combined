#!/usr/bin/env bash
set -euo pipefail

REPO_ID="jasonzhango/SPAR-7M-RGBD"
DEFAULT_OUTPUT_DIR="./source_data"

usage() {
  cat <<'USAGE'
Download SPAR-7M-RGBD from Hugging Face.

Usage:
  bash dataset/download_spar_7m_rgbd.sh [--output-dir DIR] [--extract] [--delete-archives]

Options:
  --output-dir DIR   Destination root directory. Default: ./source_data
  --extract          Extract archives after download. This needs about 184GB+ free space.
  --delete-archives  Delete downloaded .tar.gz archives after successful extraction.
  -h, --help         Show this help.

Prerequisites:
  pip install hf
  hf auth login
USAGE
}

OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
EXTRACT=0
DELETE_ARCHIVES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="${2:?--output-dir requires a directory}"
      shift 2
      ;;
    --extract)
      EXTRACT=1
      shift
      ;;
    --delete-archives)
      DELETE_ARCHIVES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing 'hf' CLI. Install it with: pip install hf" >&2
  exit 1
fi

ARCHIVE_DIR="${OUTPUT_DIR%/}/SPAR-7M-RGBD"
mkdir -p "$ARCHIVE_DIR"

echo "Downloading ${REPO_ID} archives to: ${ARCHIVE_DIR}"
hf download "$REPO_ID" \
  --repo-type dataset \
  --local-dir "$ARCHIVE_DIR" \
  --include 'spar-rgbd-*.tar.gz' 'spar-rgbd-sha256.txt'

if [[ -f "${ARCHIVE_DIR}/spar-rgbd-sha256.txt" ]]; then
  if command -v sha256sum >/dev/null 2>&1; then
    echo "Verifying SHA256 checksums..."
    (
      cd "$ARCHIVE_DIR"
      sha256sum -c spar-rgbd-sha256.txt
    )
  else
    echo "sha256sum not found; skipping checksum verification."
  fi
fi

if [[ "$EXTRACT" -eq 1 ]]; then
  echo "Extracting concatenated SPAR archives into: ${OUTPUT_DIR}"
  cat "${ARCHIVE_DIR}"/spar-rgbd-*.tar.gz | tar -xvzf - -C "$OUTPUT_DIR"
  if [[ "$DELETE_ARCHIVES" -eq 1 ]]; then
    echo "Deleting downloaded archives from: ${ARCHIVE_DIR}"
    rm -f "${ARCHIVE_DIR}"/spar-rgbd-*.tar.gz
  fi
else
  echo "Download complete. Archives are in: ${ARCHIVE_DIR}"
  echo "To extract later:"
  echo "  cat ${ARCHIVE_DIR}/spar-rgbd-*.tar.gz | tar -xvzf - -C ${OUTPUT_DIR}"
fi
