#! /usr/bin/env bash

# Usage: bash perplexity-wrapper.sh <path/to/my.ilm.gz> <path/to/my_dev_text.txt>
# compile-lm path/to/my.ilm.gz -eval=path/to/my_dev_text.txt

. ./path.sh
. ./cmd.sh

compile-lm $1 -eval=$2
