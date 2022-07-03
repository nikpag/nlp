#! /usr/bin/env bash

# Usage: bash main.sh

# Clean everything, keep important scripts only
bash scripts/clean.sh

# Download all dependencies needed from GitHub and Google Drive
bash scripts/download.sh

# Construct initial scaffold
python scripts/3.py

# Prepare speech recognition procedure for USC-TIMIT
python scripts/4.1.py

# Prepare language model
python scripts/4.2.py

# Extract acoustic characteristics
python scripts/4.3.py

# Train acoustic model and decode sentences
python scripts/4.4.py
