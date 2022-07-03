#! /usr/bin/env python

# Usage: bash scripts/download.sh
#
# This script downloads all the necessary files from:
#   - The slp-ntua/slp-labs repository
#   - The Google Drive link provided in the exercise's description

### Install "gdown" package for downloading Google Drive files via terminal
pip install gdown

### Download GitHub files

# Remove slp-labs repo if it exists
rm -rf slp-labs

# Clone slp-labs repo
git clone git@github.com:slp-ntua/slp-labs.git

# Copy helpers.py so we can use the run_cmd() function
cp slp-labs/lab1/scripts/helpers.py scripts/

# Copy mfcc.conf and timit_format_data.sh (we will need them later)
cp slp-labs/lab2/mfcc.conf ./
cp slp-labs/lab2/timit_format_data.sh scripts/

# Delete the repo (we don't need it anymore)
rm -rf slp-labs

### Download Google Drive files

# Download usc.tgz
gdown --fuzzy "https://drive.google.com/file/d/1_mIoioHMeC2HZtIbGs1LcL4kkIF696nB/view?usp=sharing"

# Unzip it
tar zxvf usc.tgz

# Copy all files from usc folder to our working directory
cp -r usc/* ./

# Delete usc.tgz folder (both zipped and unzipped)
rm -r usc/
rm usc.tgz
