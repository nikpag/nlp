#! /usr/bin/env bash

fstdraw --isymbols=phones.syms --osymbols=phones.syms --height=100 -portrait L.fst | dot -Tpng >L.png