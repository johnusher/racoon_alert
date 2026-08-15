#!/bin/zsh
# Fetch the go2rtc media-server binary (gitignored — not stored in the repo).
# macOS Apple Silicon. Pinned to the version this project was built against.
set -e
DIR="${0:A:h}"
VER="v1.9.14"
URL="https://github.com/AlexxIT/go2rtc/releases/download/${VER}/go2rtc_mac_arm64.zip"
echo "Downloading go2rtc ${VER}…"
curl -sL -o "$DIR/go2rtc.zip" "$URL"
unzip -o -q "$DIR/go2rtc.zip" -d "$DIR"
rm -f "$DIR/go2rtc.zip"
chmod +x "$DIR/go2rtc"
"$DIR/go2rtc" -version
echo "OK -> $DIR/go2rtc"
