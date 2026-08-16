#!/bin/zsh
# Download the face models for `enroll.py` / face identification (gitignored, ~38MB).
# Both ship from OpenCV's own model zoo and are run by OpenCV itself — no extra deps.
set -e
DIR="${0:A:h}"
mkdir -p "$DIR/models"
Z="https://github.com/opencv/opencv_zoo/raw/main/models"
echo "Downloading YuNet (face detection, ~230KB)…"
curl -L -o "$DIR/models/face_detection_yunet_2023mar.onnx" \
     "$Z/face_detection_yunet/face_detection_yunet_2023mar.onnx"
echo "Downloading SFace (face recognition, ~37MB)…"
curl -L -o "$DIR/models/face_recognition_sface_2021dec.onnx" \
     "$Z/face_recognition_sface/face_recognition_sface_2021dec.onnx"
echo "OK -> $DIR/models/  (now enrol somebody: detector/enroll.py add <name> live)"
