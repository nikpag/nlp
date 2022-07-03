#! /usr/bin/env python

from helpers import run_cmd
from util import l

########################################
###   4.2 - Prepare language model   ###
########################################

### 1 ###

# silence_phones.txt && optional_silence.txt
with open("data/local/dict/silence_phones.txt", "w") as f_sil_phones:
    print("sil", file=f_sil_phones)

# nonsilence_phones.txt

with open("data/local/dict/optional_silence.txt", "w") as f_opt_sil:
    print("sil", file=f_opt_sil)

# Initialise phones as a set (we don't want duplicates)
phones = set()

# Read all phones in "lexicon.txt" and add them in a file, then sort them.
with open("lexicon.txt") as f_lexicon, open("data/local/dict/nonsilence_phones.txt", "w") as f_nonsil_phones:
    for line in f_lexicon.readlines():
        line_phones = line.split("\t")[1].strip().split()
        for phone in line_phones:
            phones.add(phone)

    # We don't want the silence phone
    phones.discard("sil")

    # Sort the remaining phones
    phones = sorted(list(phones))

    # Print one phone per line
    for i, phone in enumerate(phones):
        print(phone, file=f_nonsil_phones)

# lexicon.txt: Maps every phone to itself (we decode phones, not words).
with open("data/local/dict/lexicon.txt", "w") as f_lexicon:
    temp_list = sorted((phones + ["sil"]))
    for i, phone in enumerate(temp_list):
        print(f"{phone} {phone}", file=f_lexicon)

# lm_train.text
for dir in ["dev", "test", "train"]:
    with open(f"data/local/dict/lm_{dir}.text", "w") as f_lm, open(f"data/{dir}/text") as f_text:
        for line in f_text.readlines():
            # First token of every line is the uttid, the rest is the sentence
            uttid, *sentence = line.split()

            # Add <s> and </s>
            full_line = [uttid] + ["<s>"] + sentence + ["</s>"]
            print(" ".join(full_line), file=f_lm)

# extra_questions: empty file
run_cmd("touch data/local/dict/extra_questions.txt")

run_cmd("cp ../../tools/irstlm/scripts/build-lm.sh scripts/")

### 2 ###

# Create intermediate unigram and bigram language models
for n in [1, 2]:
    ug_or_bg = "ug" if n == 1 else "bg"
    run_cmd(f"rm -f data/local/lm_tmp/lm_phone_{ug_or_bg}.ilm.gz") # Remove if exists
    print(run_cmd(f"bash scripts/build-lm-wrapper.sh data/local/dict/lm_train.text {n} data/local/lm_tmp/lm_phone_{ug_or_bg}.ilm.gz"))

### 3 ###

# Compile language model in ARPA format
for n in [1, 2]:
    ug_or_bg = "ug" if n == 1 else "bg"
    print(run_cmd(f"bash scripts/compile-lm-wrapper.sh data/local/lm_tmp/lm_phone_{ug_or_bg}.ilm.gz data/local/nist_lm/lm_phone_{ug_or_bg}.arpa.gz"))

### 4 ###

# Create language FST
print(run_cmd(f"bash scripts/prepare_lang-wrapper.sh data/local/dict data/local/lm_tmp data/lang"))

### 5 ###

# Sort wav.scp, text, utt2spk in data/{train,dev,test}
for dir in ["data/dev", "data/test", "data/train"]:
    for file_name in ["wav.scp", "text", "utt2spk"]:
        run_cmd(f"sort {dir}/{file_name} > {dir}/{file_name}~")
        run_cmd(f"mv {dir}/{file_name}~ {dir}/{file_name}")

### 6 ###

# Execute utt2spk_to_spk2utt.pl
for dir in ["data/dev", "data/test", "data/train"]:
    run_cmd(f"perl utils/utt2spk_to_spk2utt.pl {dir}/utt2spk > {dir}/spk2utt")

### Q1 ###

for dir in ["dev", "test"]:
    for ug_or_bg in ["ug", "bg"]:
        print(run_cmd(f"bash scripts/perplexity-wrapper.sh data/local/lm_tmp/lm_phone_{ug_or_bg}.ilm.gz data/local/dict/lm_{dir}"))

for dir in ["dev", "test"]:
    for x in ["ug", "bg"]:
        with open(f"perplex-{dir}-{x}", "w") as f_perplex:
            perp_str = ""
            perp_str += f"Perplexity for {dir} {x}: "
            for k, v in l[f"{dir}_{x}"].items():
                perp_str += f"{k}={v} "
            print(perp_str, file=f_perplex)

### 7 ###

# Create grammar FST. Based on the timit/ recipe.
print(run_cmd("bash scripts/timit_format_data.sh"))

print("### 4.2 done ###")
