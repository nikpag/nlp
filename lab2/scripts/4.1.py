#! /usr/bin/env python

from helpers import run_cmd

##################################################
###   4.1 - Prepare ASR Recipe for USC-TIMIT   ###
##################################################

### 1 ###

# Get path.sh and cmd.sh from Wall Street Journal recipe
run_cmd("cp ../wsj/s5/{path.sh,cmd.sh} ./")

# path.sh: Set KALDI_ROOT
with open("tmp", "w") as f_tmp, open("path.sh") as f_path:
    for i, line in enumerate(f_path.readlines()):
        if "export KALDI_ROOT" in line:
            line = "export KALDI_ROOT=`pwd`/../..\n"
        print(line, file=f_tmp, end="")

run_cmd("mv tmp path.sh")

# cmd.sh: Change train_cmd, decode_cmd and cuda_cmd to run.pl
with open("tmp", "w") as f_tmp, open("cmd.sh") as f_cmd:
    for i, line in enumerate(f_cmd.readlines()):
        if "export train_cmd" in line:
            line = "export train_cmd=run.pl\n"
        elif "export decode_cmd" in line:
            line = "export decode_cmd=run.pl\n"
        elif "export cuda_cmd" in line:
            line = "export cuda_cmd=run.pl\n"
        print(line, file=f_tmp, end="")

run_cmd("mv tmp cmd.sh")

### 2 ###

# Create steps and utils soft links
run_cmd("ln -sf ../wsj/s5/{steps,utils} ./")

### 3 ###

# Create local directory
run_cmd("mkdir -p local")

# Create score.sh soft link (points to steps/score_kaldi.sh)
run_cmd("ln -sfr steps/score_kaldi.sh local/score.sh")

### 4 ###

# Create conf folder
run_cmd("mkdir -p conf")

# Copy mfcc.conf (taken from slp-ntua/slp-labs)
run_cmd("mv mfcc.conf conf/")

### 5 ###

# Create data/lang, data/local/dict, data/local/lm_tmp, data/local/nist_lm
run_cmd("mkdir -p data/lang data/local/dict data/local/lm_tmp data/local/nist_lm")

print("### 4.1 done ###")
