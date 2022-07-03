#! /usr/bin/env python

from helpers import run_cmd
import os
import re

#########################################
###   3.4 - Create initial scaffold   ###
#########################################

# Create data, data/dev, data/test, data/train directories
run_cmd("mkdir -p data")
run_cmd("mkdir -p data/{dev,test,train}")

# Create a dictionary containing all the transcriptions found in "transcriptions.txt".
# We will use it for creating the "text" files.
with open("transcriptions.txt") as f_tran:
    lines = f_tran.read().splitlines()
    transcriptions = dict([line.split("\t") for line in lines])

# Create a dictionary containing all mappings from words to phonemes found in "lexicon.txt".
# We will also use it for creating the "text" files.
with open("lexicon.txt") as f_lex:
    lines = f_lex.read().splitlines()

    # Convert to lowercase
    lines = [line.lower() for line in lines]

    # If multiple voicings, keep one (e.g. ALICE, ALICE(1))
    lines = [re.sub("\(\d\)", "", line) for line in lines]

    lexicon = dict([line.split("\t") for line in lines])

dir_file_pairs = [
    ("train", "training.txt"),
    ("dev", "validation.txt"),
    ("test", "testing.txt"),
]

# Create necessary files
for (dir, file_name) in dir_file_pairs:
    # Create "uttids"
    run_cmd(f"cp filesets/{file_name} data/{dir}/uttids")

    with open(f"data/{dir}/uttids") as f_uttids:
        uttids = f_uttids.read().splitlines()

        # Create "utt2spk"
        with open(f"data/{dir}/utt2spk", "w") as f_utt2spk:
            speakers = [uttid.split("_")[0] for uttid in uttids]
            for uttid, speaker in zip(uttids, speakers):
                print(f"{uttid} {speaker}", file=f_utt2spk)

        # Create "wav.scp"
        with open(f"data/{dir}/wav.scp", "w") as f_wav:
            for uttid in uttids:
                wavpath = os.path.realpath(f"wav/{uttid}.wav")
                print(f"{uttid} {wavpath}", file=f_wav)

        # Create "text"
        with open(f"data/{dir}/text_orig", "w") as f_text_orig, open(f"data/{dir}/text", "w") as f_text:
            for uttid in uttids:
                sentence_id = uttid.split("_")[1]
                sentence = transcriptions[sentence_id]

                # Write original sentence to "text" file so we can use it for future reference
                print(f"{uttid} {sentence}", file=f_text_orig)

                # Convert to lowercase
                sentence = sentence.lower()

                # Transform "-" to space, since it makes more sense than just removing it
                sentence = re.sub("-", " ", sentence)
                sentence = re.sub("[^a-zA-Z' ]", "", sentence)
                phonemes = [lexicon[word] for word in sentence.split()]
                print(f"{uttid} sil{''.join(phonemes)} sil", file=f_text)



print("### 3 done ###")
