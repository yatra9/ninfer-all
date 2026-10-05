#!/bin/sh
set -eu
# Run in a WSLC container based on ninfer-all:dev, with the model volume at /models.
# Download only the explicitly built executable from the development container.
mkdir -p /acceptance /videos
python3 -c 'import sys,urllib.request; urllib.request.urlretrieve(sys.argv[1], "/acceptance/ninfer-serve")' "$1"
chmod +x /acceptance/ninfer-serve
ffmpeg -v error -f lavfi -i color=red:size=192x128:rate=2:duration=2 \
  -c:v libx264 -y /videos/red.mp4
ffmpeg -v error -i /videos/red.mp4 -c copy -y /videos/red.mkv
ffmpeg -v error -f lavfi -i color=blue:size=192x128:rate=2:duration=2 \
  -c:v libx264 -y /videos/blue.mp4
ffmpeg -v error -f lavfi -i color=red:size=192x128:rate=2:duration=1 \
  -f lavfi -i color=blue:size=192x128:rate=2:duration=1 \
  -filter_complex '[0:v][1:v]concat=n=2:v=1:a=0[v]' -map '[v]' \
  -c:v libx264 -y /videos/mystery.mp4
