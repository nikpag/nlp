#! /usr/bin/env bash

# Usage: bash prepare_lang-wrapper.sh <dict-src-dir> <tmp-dir> <lang-dir>

. ./path.sh
. ./cmd.sh

bash prepare_lang.sh $1 "<oov>" $2 $3
