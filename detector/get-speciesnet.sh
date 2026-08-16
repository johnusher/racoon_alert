#!/bin/zsh
# Download the SpeciesNet crop classifier — the second opinion on MegaDetector's class
# (gitignored, ~225MB). v4.0.1a "always_crop", the variant meant to run on detector crops.
#
# We take the released weights straight from the model repo rather than `pip install
# speciesnet`: that package needs Python <3.14 and pulls the old yolov5 this project
# already refuses (see requirements.txt). The weights are a torch.fx GraphModule that
# was converted from ONNX, so `onnx2torch` must be installed for torch.load to resolve
# the pickle — `pip install -r requirements.txt` covers it.
set -e
DIR="${0:A:h}"
mkdir -p "$DIR/models"
B="https://huggingface.co/Addax-Data-Science/SPECIESNET-v4-0-1-A-v1/resolve/main"
STEM="always_crop_99710272_22x8_v12_epoch_00148"
echo "Downloading SpeciesNet labels (2498 classes, ~250KB)…"
curl -L -o "$DIR/models/speciesnet_labels.txt" "$B/$STEM.labels.txt"
echo "Downloading SpeciesNet crop classifier (EfficientNetV2-M, ~225MB)…"
curl -L -o "$DIR/models/speciesnet_crop_4.0.1a.pt" "$B/$STEM.pt"
echo "OK -> $DIR/models/  (check it: ../.venv/bin/python detector/speciesnet.py <clip>)"
