import os
import warnings
import sys

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from config import EMB_PATH, MAX_SENTENCE_SIZE
from dataloading import SentenceDataset
from models import BaselineDNN, Model21, Model22, Model31, Model32, Model41, Model42
from training import train_dataset, eval_dataset

from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################

# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = False
BATCH_SIZE = 128

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

DATASET = "MR"

print(f"\n###### {DATASET} DATASET ######""")

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to integers
le = LabelEncoder()

y_train = le.fit_transform(y_train)  # EX1
y_test = le.fit_transform(y_test)    # EX1
n_classes = le.classes_.size         # EX1 - LabelEncoder.classes_.size

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

# EX4 - Define our PyTorch-based DataLoader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # EX7
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)                  # EX7

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################

models_dict = {
    "BaselineDNN":
        BaselineDNN(
            output_size=n_classes,  # EX8
            embeddings=embeddings,
            trainable_emb=EMB_TRAINABLE
        ),

    "Model21":
        Model21(
            output_size=n_classes,
            embeddings=embeddings,
            hidden_dim=50,
            trainable_emb=EMB_TRAINABLE
        ),

    "Model22":
        Model22(
            output_size=n_classes,
            embeddings=embeddings,
            hidden_dim=50,
            trainable_emb=EMB_TRAINABLE
        ),

    "Model31":
        Model31(
            output_size=n_classes,
            embeddings=embeddings,
            hidden_dim=50,
            attention_dim=50,
            non_linear="tanh",
            trainable_emb=EMB_TRAINABLE
        ),

    "Model32":
        Model32(
            output_size=n_classes,
            embeddings=embeddings,
            hidden_dim=50,
            attention_dim=50,
            non_linear="tanh",
            trainable_emb=EMB_TRAINABLE
        ),

    "Model41":
        Model41(
            output_size=n_classes,
            embeddings=embeddings,
            hidden_dim=50,
            trainable_emb=EMB_TRAINABLE
        ),

    "Model42":
        Model42(
            output_size=n_classes,
            embeddings=embeddings,
            hidden_dim=50,
            attention_dim=50,
            non_linear="tanh",
            trainable_emb=EMB_TRAINABLE
        ),
}

modelname = sys.argv[1] if len(sys.argv) > 1 else "BaselineDNN"
model = models_dict[modelname]

# move the mode weight to cpu or gpu
model.to(DEVICE)
print(model)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
criterion = torch.nn.BCEWithLogitsLoss() if n_classes == 2 else torch.nn.CrossEntropyLoss()  # EX8
parameters = filter(lambda p: p.requires_grad, model.parameters())                           # EX8
optimizer = torch.optim.Adam(parameters, lr=0.001)                                           # EX8

#############################################################################
# Training Pipeline
#############################################################################

train_losses = []
test_losses = []

epoch_dict = {
    "BaselineDNN": 50,
    "Model21": 20,
    "Model22": 20,
    "Model31": 100,
    "Model32": 20,
    "Model41": 20,
    "Model42": 20
}

EPOCHS = epoch_dict[modelname]

for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer, DATASET)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model,
                                                            criterion,
                                                            DATASET)

    # Append the loss (for plotting)
    train_losses.append(train_loss)

    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                        model,
                                                        criterion,
                                                        DATASET)
    # Append the loss (for plotting)
    test_losses.append(test_loss)

torch.save(model, f"models/{modelname}.pt")

for name, list in zip(["train", "test"], [train_losses, test_losses]):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    step = EPOCHS/10
    plt.xticks([i*step for i in range(11)])
    plt.plot(list)
    plt.savefig(f"results/{modelname}_{name}.pdf")

#############################################################################
# Evaluation
#############################################################################

for (golds, preds), name in zip([(y_train_gold, y_train_pred), (y_test_gold, y_test_pred)], ["train", "test"]):
    accuracy = 0
    recall = 0
    f1_score = 0

    for gold, pred in zip(golds, preds):
        accuracy += metrics.accuracy_score(gold, pred)
        recall +=  metrics.recall_score(gold, pred, average="macro")
        f1_score += metrics.f1_score(gold, pred, average="macro")

    denom = len(golds)
    accuracy /= denom
    recall /= denom
    f1_score /= denom

    # Output to console
    print(f"\n### EVALUATION OF {name.upper()} DATASET ###\n")
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")

    # Output to file
    with open(f"results/{modelname}_{name}.txt", "w") as f:
        print(f"Accuracy: {accuracy}", file=f)
        print(f"Recall: {recall}", file=f)
        print(f"F1 Score: {f1_score}", file=f)
