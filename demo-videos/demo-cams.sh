#!/bin/sh
# Camera emulator of the `demo-cams` compose profile (story 1.7).
# Publishes three synthetic cameras to MediaMTX, one per camera type (camera-profiles.md):
#   cam1  webcam    640x480,   face ~200 px, with speech audio
#   cam2  sector    1920x1080, face ~80 px,  with speech audio
#   cam3  security  2560x1440, face ~35 px,  no audio
# They are built live from bench/data/face.jpg (public domain) and speech.wav (MIT), see
# backend/bench/data/ATTRIBUTION.txt. Story 1.8 replaces them with pre-encoded openly licensed
# clips looped with -c copy. Each publisher restarts if MediaMTX goes away.
set -u
SRC=${SRC:-/src}
RTSP=${RTSP:-rtsp://mediamtx:8554}

publish() { # name w h face_px audio(yes|no)
  name=$1 w=$2 h=$3 face=$4 audio=$5
  # face.jpg is 320x320 with a face about 0.45 of its width
  img=$(( face * 100 / 45 / 2 * 2 ))
  if [ "$audio" = yes ]; then
    ain="-re -stream_loop -1 -i $SRC/speech.wav"
    aout="-map 1:a -c:a aac -b:a 64k"
  else
    ain="" aout="-an"
  fi
  while :; do
    # shellcheck disable=SC2086
    ffmpeg -hide_banner -loglevel warning -re -loop 1 -framerate 10 -i "$SRC/face.jpg" $ain \
      -map 0:v -vf "scale=-2:${img},pad=${w}:${h}:(ow-iw)/2:(oh-ih)/3:color=0x203040,format=yuv420p" \
      -r 10 -c:v libx264 -preset ultrafast -tune zerolatency -g 20 $aout \
      -f rtsp -rtsp_transport tcp "$RTSP/$name"
    echo "demo-cams: $name publisher exited, restarting in 2 s" >&2
    sleep 2
  done
}

publish cam1 640 480 200 yes &
publish cam2 1920 1080 80 yes &
publish cam3 2560 1440 35 no &
wait
