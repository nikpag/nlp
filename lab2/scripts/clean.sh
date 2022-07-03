#! /usr/bin/env bash

# Usage: bash scripts/clean.sh

# Get working directory name (not full path)
# e.g. kaldi/egs/usc --> usc
WORK_DIR=$(pwd | rev | cut -d "/" -f1 | rev)

# Create a backup of important scripts
mkdir -p ../tmp
cp -r main.sh scripts/ ../tmp/

# Remove all files
rm -r *

# Bring important scripts back
cp -r ../tmp/* ../"$WORK_DIR"

# Delete tmp directory, as it is no longer needed
rm -r ../tmp
