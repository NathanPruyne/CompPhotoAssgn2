#!/bin/bash

# Capture a burst of images at a fixed shutter speed for noise calibration (part 5).
# Usage: ./capture_noise.sh OUTPUT_DIR [SHUTTER_SPEED] [NUM_FRAMES]
#   e.g. ./capture_noise.sh ../data/noise/dark 1/60 50
# If SHUTTER_SPEED is omitted, the camera's current shutter speed is used.
# Frames already in OUTPUT_DIR are skipped, so the script can be rerun after a failure.

OUTPUT_DIR="$1"
SHUTTER_SPEED="$2"
NUM_FRAMES="${3:-50}"
MAX_RETRIES=3

if [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 OUTPUT_DIR [SHUTTER_SPEED] [NUM_FRAMES]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

gphoto2 --auto-detect

if [ -n "$SHUTTER_SPEED" ]; then
    echo "Setting shutter speed to $SHUTTER_SPEED"
    gphoto2 --set-config-value /main/capturesettings/shutterspeed="$SHUTTER_SPEED" || exit 1
fi

for i in $(seq 1 "$NUM_FRAMES"); do

    name=$(printf "frame%02d" "$i")

    if compgen -G "$OUTPUT_DIR/$name.*" > /dev/null; then
        echo "Skipping $name, already captured"
        continue
    fi

    for attempt in $(seq 1 "$MAX_RETRIES"); do
        echo "Capturing $name ($i/$NUM_FRAMES)"
        if gphoto2 --capture-image-and-download --filename "$OUTPUT_DIR/$name.%C"; then
            break
        fi

        # the D3500 can hang after a few captures, resetting the connection fixes it
        echo "Capture failed, resetting camera (attempt $attempt/$MAX_RETRIES)"
        rm -f "$OUTPUT_DIR/$name".*
        gphoto2 --reset
        sleep 2

        if [ "$attempt" -eq "$MAX_RETRIES" ]; then
            echo "Giving up on $name"
            exit 1
        fi
    done
done

echo "Captured $NUM_FRAMES frames in $OUTPUT_DIR"
