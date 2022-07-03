from helpers import run_cmd

with open("lexicon.txt") as f:
    words = [line.split()[0] for line in f.readlines()]

with open("edits.txt", "w") as f:
    for i, w1 in enumerate(words):
        for w2 in words[i:]:
            print(f"{w1} --> {w2}", file=f)
            print("======================", file=f)
            print(run_cmd(f"bash word_edits.sh {w1} {w2}"), file=f)