I_AM_PATIENT = True # Set to True in order to execute time intensive tasks, such as training the model.

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from scripts.helpers import run_cmd
import contractions
import re
import warnings; warnings.simplefilter("ignore")
import nltk

###################
###   Step 12   ###
###################

### 12a 
# From Step 1

run_cmd("python scripts/fetch_gutenberg_partB.py > data/corpus2.txt")
f = open("data/corpus2.txt")
text = f.read()

nltk.download("punkt")

tokenized_sentences = sent_tokenize(text)

# From scripts fetch gutenberg

def clean_text(s):
    s = s.strip()  # Strip leading / trailing spaces
    s = s.lower()  # Convert to lowercase
    s = contractions.fix(s)  # e.g. don't -> do not, you're -> you are
    s = re.sub("\s+", " ", s)  # Strip multiple whitespace
    s = re.sub(r"[^a-z\s]", " ", s)  # Keep only lowercase letters and spaces

    return s

cleaned_sentences = [clean_text(sentence) for sentence in tokenized_sentences]

words = [word_tokenize(sentence) for sentence in cleaned_sentences] #tokenize every word in each sentence


### 12b

if I_AM_PATIENT:
    model = Word2Vec(words, window=5, size=100, workers=8)
    model.train(words, total_examples=len(tokenized_sentences), epochs=1000)
    model.save("temp_model.model")

if I_AM_PATIENT:
    model_10 = Word2Vec(words, window=5, size=100, workers=8)
    model_10.train(words, total_examples=len(tokenized_sentences), epochs=10)
    model_10.save("temp_model2.model")

if I_AM_PATIENT:
    model_100 = Word2Vec(words, window=5, size=100, workers=8)
    model_100.train(words, total_examples=len(tokenized_sentences), epochs=100)
    model_100.save("temp_model3.model")

model = Word2Vec.load("temp_model.model")

### 12c
list_of_words = ["bible","book","bank","water"]
def similarity(model):
    for word in list_of_words:
        similar = model.wv.most_similar(word, topn=5)
        print("top 5 similar words for:",word,similar,"\n")

# Model 10 epoch
similarity(model_10)

# Model 100 epoch
similarity(model_100)

# Model 1000 epoch
similarity(model)

### 12d
list_of_words = [["girls","queen","kings"], ["good","tall","taller"], ["france","paris","london"]]
def similarity_vectors(model):
    for word1, word2, word3 in list_of_words:
        print(word1, word2, word3)
        similar = model.wv.most_similar(positive=[word1, word3],negative=[word2])
        print(f"Top 5 similar words for {word1} - {word2} + {word3} {similar}\n")

similarity_vectors(model)

### 12e

model2 = KeyedVectors.load_word2vec_format("data/GoogleNewsvectorsnegative300.bin", binary=True, limit=250000)

similarity(model2) # Google model similarity for words

similarity_vectors(model2) # Google model similarity for vectors

###################
###   Step 13   ###
###################

with open("data/embeddings.tsv", "w") as f:
    words = model.wv.vocab.keys()
    for word in words:
        vector = list(model.wv.get_vector(word))
        for vector_element in vector:
            print(vector_element, file=f, end="\t")
        print(file=f)

with open("data/metadata.tsv", "w") as f:
    words = model.wv.vocab.keys()
    for word in words:
        print(word, file=f)

###################
###   Step 14   ###
###################

from scripts.helpers import run_cmd
run_cmd("python scripts/w2v_sentiment_analysis.py")


