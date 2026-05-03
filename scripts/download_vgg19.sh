#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${ROOT_DIR}/weights"
DEST_FILE="${DEST_DIR}/vgg19-dcbb9e9d.pth"
URL="https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"

mkdir -p "${DEST_DIR}"
echo "Downloading VGG19 weights to ${DEST_FILE}"
curl -L "${URL}" -o "${DEST_FILE}"
echo "Done."
