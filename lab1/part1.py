I_AM_PATIENT = False    # Set to true in order to fully execute the script,
                        # including time-intensive tasks such as:
                        #   - Testing/evaluating the spell checkers
                        #   - Saving all edits from wikipedia to a file
                        # Keep in mind that the execution time increases greatly.
                        # Time measurements on our machines:
                        # I_AM_PATIENT = True --> ~50 min
                        # I_AM_PATIENT = False --> ~15 sec

##########################
###   Global imports   ###
##########################

from scripts.util import format_arc, EPS, CHARS, INFINITY
from scripts.helpers import run_cmd
from collections import Counter
import numpy as np

##################################
###   Step 1 - Create Corpus   ###
##################################

### 1a - Download the corpus and save it to a file
run_cmd("python scripts/fetch_gutenberg.py > data/corpus.txt")

######################################
###   Step 2 - Create dictionary   ###
######################################

### 2a - Create a {word: frequency} dictionary based on the above corpus
with open("data/corpus.txt") as f:
    dictionary = dict(Counter(f.read().split()))

### 2b - Filter tokens that show up fewer than 5 times 
filtered_dictionary = dict(filter(lambda item: item[1] >= 5, dictionary.items()))

### 2c - Create two tab-separated columns and save to vocab/words.vocab.txt
with open("vocab/words.vocab.txt", "w") as f:
    for k, v in filtered_dictionary.items():
        print(f"{k}\t{v}", file=f)

################################################
###   Step 3 - Create input/output symbols   ###
################################################

### 3a - Create function that maps lowercase chars to indices
###    - Save the mapping to vocab/chars.syms

def lowercase_to_index(c):
    return 0 if c == EPS else ord(c) # ord() returns the ASCII code

def create_lowercase_to_index_file():
    with open("vocab/chars.syms", "w") as f:
        for char in [EPS] + CHARS:
            print(f"{char}\t{lowercase_to_index(char)}", file=f) # Tab separated values

create_lowercase_to_index_file()

### 3b - Create a word-to-index file similar to the above

def create_word_to_index_file(dictionary):
    with open("vocab/words.syms", "w") as f:
        print(f"{EPS}\t0", file=f)

        index = 1
        for word in dictionary:
            print(f"{word}\t{index}", file=f)
            index += 1

create_word_to_index_file(filtered_dictionary)

####################################################
###   Step 4 - Create edit distance transducer   ###
####################################################

### 4a - Create a Levenshtein distance transducer (L)
def create_L_transducer_file(charset, filename):
    with open(filename, "w") as f:
        for c in charset:
            print(format_arc(0, 0, c, c, weight=0), file=f)   # 1: Every character to itself with weight 0 (no edit)
            print(format_arc(0, 0, c, EPS, weight=1), file=f) # 2: Every character to EPS with weight 1 (deletion)
            print(format_arc(0, 0, EPS, c, weight=1), file=f) # 3: EPS to every other character with weight 1 (insertion)
            
            for other_c in charset:                           # 4: Every character to every other character with weight 1
                if c == other_c:
                    continue # If the characters are the same the weight is 0
                print(format_arc(0, 0, c, other_c, weight=1), file=f)

        print(0, file=f)  # Accepting state

### 4b - Save fsts/L.fst file in openfst text format ###
create_L_transducer_file(CHARS, "fsts/L.fst")

### 4c - Compile L and save to fsts/L.binfst
run_cmd("fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms fsts/L.fst fsts/L.binfst")

### 4z - Use fstdraw to draw L. We use a small subset of characters for better visualization.

small_char_subset = list("ab")

create_L_transducer_file(small_char_subset, "fsts/L_small.fst")

# Compile and draw L using a small subset of characters
run_cmd("fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms fsts/L_small.fst fsts/L_small.binfst")
run_cmd("fstdraw --isymbols=vocab/chars.syms --osymbols=vocab/chars.syms -portrait fsts/L_small.binfst | dot -Tpng >fsts/L_small.png")

###############################################
###   Step 5 - Create dictionary acceptor   ###
###############################################

### 5a - Create an acceptor (V) that accepts every word of the dictionary from step 2

def create_V_acceptor_file(dictionary, filename):
    s = 1
    # Accept state number should be large enough so it doesn't mess with other states
    accept_state = sum([len(word) for word in dictionary]) + 1 

    with open(filename, "w") as f:
        for word in dictionary:
            for i, c in enumerate(word):
                if i == 0:
                    # First letter of each word.
                    # Output whole word at first letter (we will then output <eps> for subsequent letters)
                    print(format_arc(0, s, c, word, weight=0), file=f) 
                else:
                    # Rest of letters
                    print(format_arc(s, s + 1, c, EPS, weight=0), file=f)
                    s += 1

                if i == len(word) - 1:
                    # End of word. Connect with accepting state through an epsilon transition
                    print(format_arc(s, accept_state, EPS, EPS, weight=0), file=f)
            s += 1
        print(accept_state, file=f)

create_V_acceptor_file(filtered_dictionary, "fsts/V_nonoptimal.fst")

### 5b - Optimize the model
run_cmd("fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/words.syms fsts/V_nonoptimal.fst  fsts/V_nonoptimal.binfst")
run_cmd("fstrmepsilon fsts/V_nonoptimal.binfst | fstdeterminize | fstminimize >fsts/V.binfst")

### 5c - Save the acceptor file in openfst text format 
run_cmd("fstprint -isymbols=vocab/chars.syms -osymbols=vocab/words.syms fsts/V.binfst > fsts/V.fst")

### 5d - Compile the acceptor and save it to fsts/V.binfst
run_cmd("fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/words.syms fsts/V.fst fsts/V.binfst")

### 5e - Use fstdraw to draw V.
###         - We use a small subset of the words for easier visualization
###         - We draw V before any optimizations 
###         - We then apply optimizations one after the other, drawing the intermediate results

small_dictionary = {k: v for k, v in list(filtered_dictionary.items())[0:3]}
create_V_acceptor_file(small_dictionary, "fsts/V_nonoptimal_small.fst")

# Compile and optimize V. Save intermediate results as files so we can draw them later.
run_cmd("fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/words.syms fsts/V_nonoptimal_small.fst fsts/V_nonoptimal_small.binfst")
run_cmd("fstrmepsilon fsts/V_nonoptimal_small.binfst fsts/V_rmepsilon_small.binfst")
run_cmd("fstdeterminize fsts/V_rmepsilon_small.binfst fsts/V_rmepsilon_determinized_small.binfst")
run_cmd("fstminimize fsts/V_rmepsilon_determinized_small.binfst fsts/V_optimal_small.binfst")

# Draw V before any optimizations, as well as in between steps (rmepsilon, determinize, minimize)
for fstname in ["V_nonoptimal_small", "V_rmepsilon_small", "V_rmepsilon_determinized_small", "V_optimal_small"]:
    run_cmd(f"fstdraw --isymbols=vocab/chars.syms --osymbols=vocab/words.syms -portrait fsts/{fstname}.binfst | dot -Tpng >fsts/{fstname}.png")

#########################################
###   Step 6 - Create spell checker   ###
#########################################

### 6a - Compose L with V to produce S 

def compose_fsts(F, G, H):
    """Composes F with G to produce H = FG. 
    
    Calls fstarcsort on F's outputs and G's inputs to ensure proper composition.

    Args:
        F (string): Name of the first FST binary (without .binfst extension)
        G (string): Name of the second FST binary (without .binfst extension)
        H (string): Name for the composed FST binary (without .binfst extension)
    """
    run_cmd(f"fstarcsort --sort_type=olabel fsts/{F}.binfst fsts/{F}_sorted_outputs.binfst")
    run_cmd(f"fstarcsort --sort_type=ilabel fsts/{G}.binfst fsts/{G}_sorted_inputs.binfst")
    run_cmd(f"fstcompose fsts/{F}_sorted_outputs.binfst fsts/{G}_sorted_inputs.binfst fsts/{H}.binfst")

compose_fsts("L", "V", "S")

#######################################
###   Step 7 - Test spell checker   ###
#######################################

### 7b - Test spell checker with 20 first words of test set

def test_spell_checker(test_file, spellchecker_binary, output_file="/dev/stdout"):
    """Tests the spell checker, using the first 20 words of the given test set

    Args:
        test_file (string): Filename of test set
        spellchecker_binary (string): Filename of spellchecker compiled FST binary
        output_file (string, optional): Filename to write test results to. Defaults to /dev/stdout.
    """
    with open(test_file) as fin, open(output_file, "w") as fout:
        for _ in range(20):
            words = fin.readline().strip().split(" ") # Get all words in a line
            correct_word = words[0][:-1] # Throw away the colon (e.g. contended: --> contended)
            wrong_words = words[1:] # The rest of the words in a line are wrong
            for wrong_word in wrong_words:
                corrected_word = run_cmd(f"bash scripts/predict.sh {spellchecker_binary} {wrong_word}") # Use predict.sh to correct the word
                print(f"{wrong_word} --> {corrected_word}, correct: {correct_word}", file=fout)

if I_AM_PATIENT:
    test_spell_checker("data/spell_test.txt", "fsts/S.binfst", "command_outputs/S_test.txt")

#########################################
###   Step 8 - Calculate edits cost   ###
#########################################

### 8a - Use only one example (abandonned --> abandoned). 
###    - Create an acceptor for the wrong word (M)
###    - Create a transducer for the correct word (N)
### 8b - Compose M with L to produce ML, then compose ML with N to produce MLN
### 8c - Execute fstshortestpath followed by fstprint

### Steps 8a-8c are handled by the word_edits.sh script
run_cmd("bash scripts/word_edits.sh abandonned abandoned")

### 8d - Create a python script that executes the above process for every line in data/wiki.txt
###    - All edits are saved to data/edits.txt

def save_all_edits_to_file():
    # We deleted 3 words that had obscure characters from data/wiki.txt
    with open("data/wiki.txt", "r") as wiki, open("data/edits.txt", "w") as edits:
        for line in wiki:
            wrong, correct = tuple(line.strip().split("\t"))
            output_line = run_cmd(f"bash scripts/word_edits.sh {wrong} {correct}")
            print(output_line, file=edits, end="")

if I_AM_PATIENT:
    save_all_edits_to_file()

### 8e - Extract the frequency of each edit to a dictionary 
###    - key = (source, target), value = frequency

def extract_frequency_to_dict(filename):
    with open(filename) as f:
        return dict(Counter([tuple(line.strip().split("\t")) for line in f.readlines()]))

edit_freq_dict = extract_frequency_to_dict("data/edits.txt")

### 8st: Repeat Step 4 for -log(frequency) weights

total_number_of_edits = sum(v for _, v in edit_freq_dict.items()) 

log_edit_freq_dict = {(src, dst): -np.log(freq / total_number_of_edits) for ((src, dst), freq) in edit_freq_dict.items()}

###### Repeat step 4a, but now create transducer E instead of L

def create_E_transducer_file(charset, filename, dictionary):
    with open(filename, "w") as f:
        for c in charset:
            # 1: Every character to itself with weight 0 (no edit)
            print(format_arc(0, 0, c, c, weight=0), file=f)

            # 2: Every character to EPS with weight -log(frequency)
            key = (c, EPS)
            w = dictionary[key] if key in dictionary else INFINITY
            print(format_arc(0, 0, c, EPS, weight=w), file=f)

            # 3: EPS to every other character with weight -log(frequency)
            key = (EPS, c)
            w = dictionary[key] if key in dictionary else INFINITY
            print(format_arc(0, 0, EPS, c, weight=w), file=f)

            # 4: Every other character to every other character with weight -log(frequency)
            for other_c in charset:
                if c == other_c:
                    continue
                key = (c, other_c)
                w = dictionary[key] if key in dictionary else INFINITY
                print(format_arc(0, 0, c, other_c, weight=w), file=f)
        print(0, file=f)  # Accepting state

###### Repeat step 4b - Save fsts/E.fst in openfst text format
create_E_transducer_file(CHARS, "fsts/E.fst", log_edit_freq_dict)

###### Repeat step 4c - Compile fsts/E.fst and save the result to fsts/E.binfst
run_cmd("fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms fsts/E.fst fsts/E.binfst")

###### Repeat step 4z - Use fstdraw to draw E. We use a small subset of characters for better visualization. 
small_char_subset = list("abc")
create_E_transducer_file(small_char_subset, "fsts/E_small.fst", log_edit_freq_dict)

# Compile and draw E using a small subset of characters
run_cmd("fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms fsts/E_small.fst fsts/E_small.binfst")
run_cmd("fstdraw --isymbols=vocab/chars.syms --osymbols=vocab/chars.syms -portrait fsts/E_small.binfst | dot -Tpng >fsts/E_small.png")

### 8z: Repeat steps 6 and 7 using E instead of L

###### Repeat step 6a - Compose E with V to produce EV 
compose_fsts("E", "V", "EV")

###### Repeat step 7b - Test spell checker with 20 first words of test set
if I_AM_PATIENT:
    test_spell_checker("data/spell_test.txt", "fsts/EV.binfst", "command_outputs/EV_test.txt")

##################################################################
###   Step 9 - Introduce word frequency (Unigram word model)   ###
##################################################################

### 9b - Create an acceptor that maps every word to itself with -log(frequency) weights

def create_W_acceptor_file(dictionary, filename):
    with open(filename, "w") as f:
        total_occurrences = sum(v for _, v in dictionary.items())
        for word, occurrences in dictionary.items():
            w = -np.log(occurrences / total_occurrences)
            print(format_arc(0, 0, word, word, w), file=f)
        print(0, file=f)

create_W_acceptor_file(filtered_dictionary, "fsts/W.fst")

### 9c - Create LVW

# Compile and optimize W
run_cmd("fstcompile -isymbols=vocab/words.syms -osymbols=vocab/words.syms fsts/W.fst fsts/W.binfst")
run_cmd("fstrmepsilon fsts/W.binfst | fstdeterminize | fstminimize >fsts/W_optimal.binfst")

# Compose L with V to produce LV, and then with W to produce LVW
compose_fsts("L", "V", "LV")
compose_fsts("LV", "W_optimal", "LVW")

### 9d - Create EVW

# Compose E with V to produce EV, and then with W to produce EVW
compose_fsts("E", "V", "EV")
compose_fsts("EV", "W_optimal", "EVW")

### 9e - Evaluate LVW
if I_AM_PATIENT:
    test_spell_checker("data/spell_test.txt", "fsts/LVW.binfst", "command_outputs/LVW_test.txt")

### 9z - Use fstdraw to draw W and VW. We use a small subset of the words for easier visualization.
small_filtered_dictionary = {k: v for k, v in list(filtered_dictionary.items())[0:3]}

# Create V and W openfst text format files using a small subset of the words
create_V_acceptor_file(small_filtered_dictionary, "fsts/V_small.fst")
create_W_acceptor_file(small_filtered_dictionary, "fsts/W_small.fst")

# Compile V and W 
run_cmd("fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/words.syms fsts/V_small.fst fsts/V_small.binfst")
run_cmd("fstcompile -isymbols=vocab/words.syms -osymbols=vocab/words.syms fsts/W_small.fst fsts/W_small.binfst")

# Compose V with W to produce VW
compose_fsts("V_small", "W_small", "VW_small")

# Draw the resulting FSTs using a small subset of the words
run_cmd("fstdraw --isymbols=vocab/words.syms --osymbols=vocab/words.syms -portrait fsts/W_small.binfst | dot -Tpng >fsts/W.png")
run_cmd("fstdraw --isymbols=vocab/chars.syms --osymbols=vocab/words.syms -portrait fsts/VW_small.binfst | dot -Tpng >fsts/VW.png")

#############################################
###   Step 10 - Evaluate spell checkers   ###
#############################################

# Evaluate the spell checkers using 
if I_AM_PATIENT:
    for spell_checker in ["LV", "LVW", "EV", "EVW"]:
        # Measured accuracies --> LV: ~0.596, LVW: ~0.044, EV: ~0.689, EVW: ~0.641
        run_cmd("python scripts/run_evaluation.py fsts/{spell_checker}.binfst > command_outputs/{spell_checker}_evaluation.txt")
