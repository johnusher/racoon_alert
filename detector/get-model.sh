#!/bin/zsh
# Download the MegaDetector v6 weights (gitignored — ~50MB). Loaded directly via ultralytics.
set -e
DIR="${0:A:h}"
mkdir -p "$DIR/models"
URL="https://zenodo.org/records/15398270/files/MDV6-yolov9-c.pt?download=1"
echo "Downloading MegaDetector v6 (yolov9-c)…"
curl -L -o "$DIR/models/MDV6-yolov9-c.pt" "$URL"
echo "OK -> $DIR/models/MDV6-yolov9-c.pt"
