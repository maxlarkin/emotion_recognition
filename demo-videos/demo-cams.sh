#!/bin/sh
# Camera emulator of the `demo-cams` compose profile.
# With prepared clips (demo-videos/clips/cam01..cam20.mp4, see backend/scripts/prep_demo_clips.py)
# it loops each to rtsp://mediamtx:8554/camN with -c copy: no re-encoding at runtime.
# Without clips it falls back to three synthetic cameras built live from bench/data (story 1.7).
set -u
SRC=${SRC:-/src}
CLIPS=${CLIPS:-/demo/clips}
RTSP=${RTSP:-rtsp://mediamtx:8554}

loop_clip() { # name file: a missing file is logged and retried, the other cameras keep running
  while :; do
    if [ -f "$2" ]; then
      ffmpeg -hide_banner -loglevel error -re -stream_loop -1 -i "$2" -c copy -f rtsp -rtsp_transport tcp "$RTSP/$1"
      echo "demo-cams: $1 publisher exited, restarting in 2 s" >&2
    else
      echo "demo-cams: $1: $2 is missing, retrying in 10 s" >&2
      sleep 8
    fi
    sleep 2
  done
}

synthetic() { # name w h face_px audio(yes|no)
  name=$1 w=$2 h=$3 face=$4 audio=$5
  img=$(( face * 100 / 45 / 2 * 2 ))  # face.jpg is 320x320 with a face about 0.45 of its width
  if [ "$audio" = yes ]; then
    ain="-re -stream_loop -1 -i $SRC/speech.wav" aout="-map 1:a -c:a aac -b:a 64k"
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

if ls "$CLIPS"/cam[0-9][0-9].mp4 >/dev/null 2>&1; then
  for n in $(seq 1 20); do
    loop_clip "cam$n" "$(printf '%s/cam%02d.mp4' "$CLIPS" "$n")" &
  done
  echo "demo-cams: publishing cam1..cam20 from $CLIPS ($(ls "$CLIPS"/cam[0-9][0-9].mp4 | wc -l) clips present)" >&2
else
  echo "demo-cams: no clips in $CLIPS (run backend/scripts/prep_demo_clips.py, see AGENTS.md); using 3 synthetic cameras" >&2
  synthetic cam1 640 480 200 yes &
  synthetic cam2 1920 1080 80 yes &
  synthetic cam3 2560 1440 35 no &
fi
wait
