#! /usr/bin/env python

from helpers import run_cmd

############################################################
###   4.4 - Train acoustic models and decode sentences   ###
############################################################

print("### 4.4 begins... ###")


### 1 ###

# Train monophone GMM-HMM acoustic model on train data
print(run_cmd("bash steps/train_mono.sh data/train data/lang exp/mono"))


### 2 ###

# Create HCLG graph
for ug_or_bg in ["ug", "bg"]:
    print(run_cmd(f"bash utils/mkgraph.sh data/lang_phones_{ug_or_bg} exp/mono exp/mono_graph_{ug_or_bg}")) # TODO why lang_phones_{} instead of lang?


### 3 ###

# Decode using Viterbi algorithm
for ug_or_bg in ["ug", "bg"]:
    for dir in ["dev", "test"]:
        print(run_cmd(f"bash steps/decode.sh exp/mono_graph_{ug_or_bg} data/{dir} exp/mono/decode_{dir}_{ug_or_bg}"))


### 5 ###

# Align phones using monophone model
print(run_cmd("bash steps/align_si.sh data/train data/lang exp/mono exp/mono_ali"))


# Train triphone model
print(run_cmd("bash steps/train_deltas.sh 2000 10000 data/train data/lang exp/mono_ali exp/tri1"))


# Create HCLG graph (again)
for ug_or_bg in ["ug", "bg"]:
    print(run_cmd(f"bash utils/mkgraph.sh data/lang_phones_{ug_or_bg} exp/tri1 exp/tri1_graph_{ug_or_bg}"))


# Decode (again)
for ug_or_bg in ["ug", "bg"]:
    for dir in ["dev", "test"]:
        print(run_cmd(f"bash steps/decode.sh exp/tri1_graph_{ug_or_bg} data/{dir} exp/tri1/decode_{dir}_{ug_or_bg}"))


print("### 4.4 done ###")
