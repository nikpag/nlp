import torch
from torch import nn
from attention import SelfAttention

# Naming convention:
# Modelxy ==> Question x.y
# For example, Model21 corresponds to question 2.1

class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__()

        # 1 - define the embedding layer
        num_embeddings, embedding_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim) # EX4

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings)) # EX4

        # 3 - define if the embedding layer will be frozen or finetuned
        self.embedding_layer.weight.requires_grad = trainable_emb  # EX4

        # 4 - define a non-linear transformation of the representations
        self.linear = nn.Linear(embedding_dim, 50)
        self.relu = nn.ReLU()  # EX5

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        self.output = nn.Linear(50, output_size)

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """

        # 1 - embed the words, using the embedding layer
        embeddings = self.embedding_layer(x)                       # EX6

        # 2 - construct a sentence representation out of the word embeddings
        representations = torch.sum(embeddings, dim=1)

        for i in range(lengths.shape[0]):
            representations[i] = representations[i] / lengths[i]   # EX6

        # 3 - transform the representations to new ones.
        representations = self.relu(self.linear(representations))  # EX6

        # 4 - project the representations to classes using a linear layer
        logits = self.output(representations)                      # EX6

        return logits

class Model21(nn.Module):
    def __init__(self, output_size, embeddings, hidden_dim, trainable_emb=False):
        super(Model21, self).__init__()

        self.hidden_dim = hidden_dim

        # Same as BaselineDNN
        num_embeddings, embedding_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings))
        self.embedding_layer.weight.requires_grad = trainable_emb

        # LSTM: e_i ==> h_i
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_size)

    def forward(self, x, lengths):
        batch_size, max_length = x.shape
        embeddings = self.embedding_layer(x)

        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)

        # The sentence representation is the final hidden state of the model
        representations = torch.zeros(batch_size, self.hidden_dim).float()

        for i in range(lengths.shape[0]):
            # This discards zeroes from padding (at the end)
            last = lengths[i] - 1 if lengths[i] <= max_length else max_length - 1
            representations[i] = ht[i, last, :]

        logits = self.linear(representations)

        return logits

class Model22(nn.Module):
    def __init__(self, output_size, embeddings, hidden_dim, trainable_emb=False):
        super(Model22, self).__init__()

        self.hidden_dim = hidden_dim

        # Same as BaselineDNN
        num_embeddings, embedding_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings))
        self.embedding_layer.weight.requires_grad = trainable_emb

        # Same as Model21
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # Careful: 1st parameter is now (3 * hidden_dim) because of different representation
        self.linear = nn.Linear(3 * hidden_dim, output_size)

    def forward(self, x, lengths):
        batch_size, max_length = x.shape
        embeddings = self.embedding_layer(x)

        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)

        # The sentence representation is the final hidden state of the model
        representations = torch.zeros(batch_size, self.hidden_dim).float()
        mean_pooling = torch.sum(ht, dim=1)

        for i in range(lengths.shape[0]):
            # This discards zeroes from padding (at the end)
            last = lengths[i] - 1 if lengths[i] <= max_length else max_length - 1
            representations[i] = ht[i, last, :]

            # Mean pooling for outputs from LSTM
            mean_pooling[i] /= lengths[i]

        # Max pooling for outputs from LSTM
        h = torch.transpose(ht, 1, 2) # So we have [N, L, C] ==> [N, C, L]
        m = nn.MaxPool1d(max_length)
        max_pooling = m(h)
        max_pooling = max_pooling.squeeze()

        # Concatenate the three parts
        representations = torch.cat((representations, mean_pooling, max_pooling), 1)

        logits = self.linear(representations)

        return logits

class Model31(nn.Module):
    def __init__(self, output_size, embeddings, hidden_dim, attention_dim, non_linear, trainable_emb=False):
        super(Model31, self).__init__()
        self.hidden_dim = hidden_dim

        # Same as BaselineDNN
        num_embeddings, embedding_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings))
        self.embedding_layer.weight.requires_grad = trainable_emb

        # Same as above
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # Attention implementation by cbaziotis
        self.attention = SelfAttention(attention_dim, non_linear)

        # Same as above
        self.linear = nn.Linear(attention_dim, output_size)

    def forward(self, x, lengths):
        batch_size, max_length = x.shape
        embeddings = self.embedding_layer(x)

        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)

        # Use the Attention mechanism defined above in order to get the sentence representation
        # Here, attention on embeddings (3.1)
        representations, _ = self.attention(embeddings, lengths)

        logits = self.linear(representations)

        return logits

class Model32(nn.Module):
    def __init__(self, output_size, embeddings, hidden_dim, attention_dim, non_linear, trainable_emb=False):
        super(Model32, self).__init__()
        self.hidden_dim = hidden_dim

        # Same as BaselineDNN
        num_embeddings, embedding_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings))
        self.embedding_layer.weight.requires_grad = trainable_emb

        # Same as above
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # Implementation by cbaziotis
        self.attention = SelfAttention(attention_dim, non_linear)

        # Same as above
        self.linear = nn.Linear(attention_dim, output_size)

    def forward(self, x, lengths):
        batch_size, max_length = x.shape
        embeddings = self.embedding_layer(x)

        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)

        # Use the Attention mechanism defined above in order to get the sentence representation
        # Here, attention on hidden states (3.2)
        representations, _ = self.attention(ht, lengths)

        logits = self.linear(representations)

        return logits


# Bidirectional
class Model41(nn.Module):
    def __init__(self, output_size, embeddings, hidden_dim, trainable_emb=False):
        super(Model41, self).__init__()
        self.hidden_dim = hidden_dim

        # Same as BaselineDNN
        num_embeddings, embedding_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings))
        self.embedding_layer.weight.requires_grad = trainable_emb

        # Same as above
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Again, careful: Bidirectionality now enforces (6 * hidden_dim)
        self.linear = nn.Linear(6 * hidden_dim, output_size)

    def forward(self, x, lengths):
        batch_size, max_length = x.shape
        embeddings = self.embedding_layer(x)

        # Same as before
        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)

        # Representation changes because of bidirectionality
        first = torch.zeros(batch_size, self.hidden_dim).float()
        second = torch.zeros(batch_size, self.hidden_dim).float()

        # "mean_pooling" needs two elements now
        mean_pooling = [
            torch.sum(ht[:, :, :self.hidden_dim], dim=1),
            torch.sum(ht[:, :, self.hidden_dim:], dim=1)
        ]

        for i in range(lengths.shape[0]):
            # This discards zeroes from padding (at the end)
            # "last" needs two elements now
            last = [
                lengths[i] - 1 if lengths[i] <= max_length else max_length - 1,
                lengths[i] - 1 if lengths[-(i+1)] <= max_length else max_length - 1
            ]

            first[i] = ht[i, last[0], :self.hidden_dim]
            second[i] = ht[-(i+1), last[1], self.hidden_dim:]

            mean_pooling[0][i] /= lengths[i]
            mean_pooling[1][i] /=  lengths[-(i+1)]

        h = [
            torch.transpose(ht[:, :, :self.hidden_dim], 1, 2), # Tranpose works like before
            torch.transpose(ht[:, :, self.hidden_dim:], 1, 2)
        ]

        m = nn.MaxPool1d(max_length)

        max_pooling = [m(item).squeeze() for item in h]

        representations = [
            torch.cat((first, mean_pooling[0], max_pooling[0]), 1),
            torch.cat((second, mean_pooling[1], max_pooling[1]), 1)
        ]

        representations_combined = torch.cat((representations[0], representations[1]), 1)

        logits = self.linear(representations_combined)

        return logits

# Bidirectional
class Model42(nn.Module):
    def __init__(self, output_size, embeddings, hidden_dim, attention_dim, non_linear, trainable_emb=False):
        super(Model42, self).__init__()
        self.hidden_dim = hidden_dim

        # Same as BaselineDNN
        num_embeddings, embedding_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings))
        self.embedding_layer.weight.requires_grad = trainable_emb

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        # Implementation by cbaziotis
        self.attention = SelfAttention(attention_dim, non_linear)

        self.linear = nn.Linear(attention_dim, output_size)

    def forward(self, x, lengths):
        batch_size, max_length = x.shape
        embeddings = self.embedding_layer(x)

        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)

        # Same as 3.2, attention on hidden states
        representations, att_scores = self.attention(ht, lengths)

        logits = self.linear(representations)

        return logits
