#!/bin/bash

DOWNLOAD_PATH="./weights/vae/kl16.ckpt"
URL="https://www.dropbox.com/scl/fi/hhmuvaiacrarfg28qxhwz/kl16.ckpt?rlkey=l44xipsezc8atcffdp4q7mwmh&dl=1"

mkdir -p "$(dirname "$DOWNLOAD_PATH")"

if [ ! -f "$DOWNLOAD_PATH" ] || [ "$1" == "--overwrite" ]; then
    echo "Downloading KL-16 VAE..."
    curl -L -o "$DOWNLOAD_PATH" "$URL"
else
    echo "File already exists at $DOWNLOAD_PATH. Use --overwrite to re-download."
fi
