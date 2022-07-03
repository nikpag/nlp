#! /usr/bin/env python
from helpers import run_cmd
from util import format_arc

phones = [
    "<eps>",
    "eh",
    "er",
    "iy",
    "m",
    "n",
    "p",
    "s",
    "t",
    "uh",
    "uw",
    "z",
]

with open("phones.syms", "w") as f:
    for i, phone in enumerate(phones):
        print(f"{phone} {i}", file=f)

with open("L.txt", "w") as f:
    for p1 in phones:
        for p2 in phones:
            # <eps> <eps> not useful
            if p1 == "<eps>" and p2 == "<eps>":
                continue

            # Same
            if p1 == p2:
                cost = 0.0

            # Insertion
            elif p1 == "<eps>":
                cost = 1.2

            # Deletion
            elif p2 == "<eps>":
                cost = 1.2

            # Substitution
            else:
                cost = 1.0
                if (p1, p2) in {
                    ("uh", "uw"), ("uw", "uh"),
                    ("er", "eh"), ("eh", "er"),
                    ("p", "t"), ("t", "p"),
                    ("m", "n"), ("n", "m"),
                    ("s", "z"), ("z", "s"),
                }:
                    cost = 0.5

            print(format_arc(0, 0, p1, p2, cost), file=f)

    print(0, file=f)

run_cmd("fstcompile --isymbols=phones.syms --osymbols=phones.syms L.txt L.fst")

with open("lexicon.txt") as f:
    words = [line.split()[0] for line in f.readlines()]

with open("edits.txt", "w") as f:
    for i, w1 in enumerate(words):
        for w2 in words[i:]:
            print(f"{w1} - {w2}", file=f)
            print(run_cmd(f"bash word_edits.sh {w1} {w2}"), file=f)



