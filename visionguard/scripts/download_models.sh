#!/usr/bin/env bash
# download_models.sh
# -------------------
# Fetches the pretrained MobileNet-SSD (Caffe) weights used by
# src/detection_module/object_detector.py. Run this once after cloning
# the repository and before using the object-detection feature.
#
# Usage:
#   bash scripts/download_models.sh

set -euo pipefail

DEST_DIR="$(dirname "$0")/../src/detection_module/models"
mkdir -p "$DEST_DIR"

PROTOTXT_URL="https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
WEIGHTS_URL="https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"

echo "Downloading MobileNet-SSD prototxt..."
curl -L "$PROTOTXT_URL" -o "$DEST_DIR/MobileNetSSD_deploy.prototxt"

echo "Downloading MobileNet-SSD weights..."
echo "(If this link is unavailable, download 'MobileNetSSD_deploy.caffemodel'"
echo " manually from the MobileNet-SSD GitHub repo/releases and place it in:"
echo " $DEST_DIR)"
curl -L "$WEIGHTS_URL" -o "$DEST_DIR/MobileNetSSD_deploy.caffemodel"

echo "Done. Model files saved to $DEST_DIR"
