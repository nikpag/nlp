#! /usr/bin/env bash

# Usage: bash scripts/build-lm-wrapper.sh <input file> <n> <output file>

. ./path.sh

bash scripts/build-lm.sh -i $1 -n $2 -o $3
