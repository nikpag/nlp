from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import glob
import numpy as np
import os
import os
import re
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression

SCRIPT_DIRECTORY = os.path.dirname(__file__)

data_dir = os.path.join(SCRIPT_DIRECTORY, "../data/aclImdb/")
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
pos_train_dir = os.path.join(train_dir, "pos")
neg_train_dir = os.path.join(train_dir, "neg")
pos_test_dir = os.path.join(test_dir, "pos")
neg_test_dir = os.path.join(test_dir, "neg")

# For memory limitations. These parameters fit in 8GB of RAM.
# If you have 16G of RAM you can experiment with the full dataset / W2V
MAX_NUM_SAMPLES = 5000
# Load first 1M word embeddings. This works because GoogleNews are roughly
# sorted from most frequent to least frequent.
# It may yield much worse results for other embeddings corpora
NUM_W2V_TO_LOAD = 250000


SEED = 42

# Fix numpy random seed for reproducibility
np.random.seed(SEED)


def strip_punctuation(s):
    return re.sub(r"[^a-zA-Z\s]", " ", s)


def preprocess(s):
    return re.sub("\s+", " ", strip_punctuation(s).lower())


def tokenize(s):
    return s.split(" ")


def preproc_tok(s):
    return tokenize(preprocess(s))


def read_samples(folder, preprocess=lambda x: x):
    samples = glob.iglob(os.path.join(folder, "*.txt"))
    data = []

    for i, sample in enumerate(samples):
        if MAX_NUM_SAMPLES > 0 and i == MAX_NUM_SAMPLES:
            break
        with open(sample, "r") as fd:
            x = [preprocess(l) for l in fd][0]
            data.append(x)

    return data


def create_corpus(pos, neg):
    corpus = np.array(pos + neg)
    y = np.array([1 for _ in pos] + [0 for _ in neg])
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)

    return list(corpus[indices]), list(y[indices])


def extract_nbow(corpus, vec_size, model_name):

    nbow = []
    for corp in corpus:
        
        counts = 0
        init_vec = np.zeros(vec_size)
        index2word_set = set(model_name.wv.index2word)

        for word in corp:
            counts = counts + 1

            if word in index2word_set:

                init_vec = np.add(init_vec, model_name[word])

        init_vec = np.divide(init_vec, counts)
        nbow.append(init_vec)

    return nbow


def train_sentiment_analysis(train_corpus, train_labels):

    regression_model = LogisticRegression()

    regression_model.fit(train_corpus, train_labels)

    return regression_model

def evaluate_sentiment_analysis(classifier, test_corpus, test_labels, model_name):
    
    accuracy = sklearn.metrics.accuracy_score(test_labels, classifier.predict(test_corpus))
    print("The accuracy of " + str(model_name) + " train model is ", accuracy)

    return accuracy


if __name__ == "__main__":
    model = Word2Vec.load('temp_model.model')
    # read train data
    pos_train = read_samples(pos_train_dir, preproc_tok)
    neg_train = read_samples(neg_train_dir, preproc_tok)
    # Read test data
    pos_test = read_samples(pos_test_dir, preproc_tok)
    neg_test = read_samples(neg_test_dir, preproc_tok) 

    # create train corpus
    train_corpus, train_labels = create_corpus(pos_train, neg_train)
    nbow_train_corpus = extract_nbow(train_corpus, 100, model)
    
    log_reg = train_sentiment_analysis(nbow_train_corpus, train_labels)
       
    test_corpus, test_labels = create_corpus(pos_test, neg_test)
    nbow_test_corpus = extract_nbow(test_corpus, 100, model)
    
    acc_log_reg = evaluate_sentiment_analysis(log_reg, nbow_test_corpus, test_labels, "our model")

    NUM_W2V_TO_LOAD = 250000
    google_model = KeyedVectors.load_word2vec_format('data/GoogleNewsvectorsnegative300.bin', binary=True, limit=NUM_W2V_TO_LOAD)
    
    nbow_corpus_google = extract_nbow(train_corpus, 300, google_model)
    
    log_reg_google = train_sentiment_analysis(nbow_corpus_google, train_labels)
    
    nbow_test_corpus_google = extract_nbow(test_corpus, 300, google_model)
    
    evaluate_sentiment_analysis(log_reg_google, nbow_test_corpus_google, test_labels, "google news model")

