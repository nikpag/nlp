#! /usr/bin/env python

import os

with open("lexicon.txt") as f:
	lexicon = {line.split()[0]: line.split()[1:] for line in f.readlines()}

for k, v in lexicon.items():
	print(k, v)

def arc(src, dst, sin, sout, cost=0):
	return f"{src} {dst} {sin} {sout} {cost}"

glob = 0
final = 10000

with open("L.txt", "w") as f:
	for word, phones in lexicon.items():
		for i, phone in enumerate(phones):
			N = len(phones)
			src = 0 if i == 0 else glob
			dst = final if i == N - 1 else glob + 1
			sin = phone
			sout = word if i == 0 else "<eps>"
			print(arc(src, dst, sin, sout), file=f)
			glob += 1
	print(final, file=f)

os.system("fstcompile --isymbols=syms --osymbols=syms L.txt L.fst")
os.system("fstdraw -isymbols=syms -osymbols=syms -portrait L.fst | dot -Tpdf > L.pdf")
