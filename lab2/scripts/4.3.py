#! /usr/bin/env python

from helpers import run_cmd

##################################################
###   4.3 - Extract acoustic characteristics   ###
##################################################



for dir in ["train", "dev", "test"]:
    print(run_cmd(f"bash steps/make_mfcc.sh data/{dir}"))

for dir in ["train", "dev", "test"]:
    print(run_cmd(f"bash steps/compute_cmvn_stats.sh data/{dir}"))

print("### 4.3 done ###")
