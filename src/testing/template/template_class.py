from random import random

import torch
import torch.nn as nn

"""
* Model : torch.nn.Module
* Focus on forward
  - Model 3 Status : Structure, Code, and Math
"""


class TemplateModel(nn.Module):
    """ Model Template"""
    def __int__(self):
        super(TemplateModel, self).__init__()
        # TODO

    def forward(self):
        # TODO
        pass


## RNN #################################################################################
class Template1RNN(nn.Module):
    """ RNN Template"""

    def __init__(self, word_count, embedding_size, hidden_size, output_size):
        super(Template1RNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(word_count, embedding_size)
        self.i2s = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(embedding_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden):
        word_vector = self.embedding(input_tensor)
        combined = torch.cat((word_vector, hidden), 1)
        hidden = self.i2s(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class Template2RNN(nn.Module):
    """ RNN Template"""

    def __init__(self, word_count: int, embedding_size: int, hidden_size: int, output_size: int, num_layers: int):
        super(Template2RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(word_count, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, bidirectional=False, batch_first=True)
        self.cls = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, input_sentence: torch.Tensor) -> torch.Tensor:
        sentence_vector = self.embedding(input_sentence)
        # get output from (hidden, output)
        output = self.rnn(sentence_vector)[0][0][len(input_sentence) - 1]
        output = self.cls(output)
        output = self.softmax(output)
        return output


## Se12Seq #################################################################################
class Encoder(nn.Module):
    def __init__(self, input_size: int, embedding_size: int, hidden_size: int, layer_size: int, dropout_ratio: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, layer_size, dropout=dropout_ratio)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input_tensor):
        # input_tensor = [input_size, batch_size]
        embedded = self.dropout(self.embedding(input_tensor))
        # embedded = [input_size, batch_size, embedding_size]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [input_size, batch_size, hidden_size * n directions]
        # hidden = [layer_size * n directions, batch_size, hidden_size]
        # cell = [layer_size * n directions, batch_size, hidden_size]
        # outputs are always from the top hidden layer
        return hidden, cell








class TemplateSeq2Seq(nn.Module):
    def __init__(self, input_word_count, output_word_count, encode_dim, decode_dim, hidden_dim, n_layers, encode_dropout, decode_dropout, device):
        super().__init__()
        self.encoder = nn.Encoder(input_word_count, encode_dim, hidden_dim, n_layers, encode_dropout)
        self.decoder = nn.Decoder(output_word_count, decode_dim, hidden_dim, n_layers, decode_dropout)
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


## Dataset #################################################################################
class Example():
    def __init__(self, data, label):
        self.data = data
        self.label = label


class TemplateDataset(torch.util.data.Dataset):
    """ 資料集 """
    def __int__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        return example.data, example.label


def collate_fun(batch):
    tensor_data = []
    tensor_labels = []
    for item, label in batch:
        tensor_data.append(torch.tensor(item))
        tensor_labels.append(torch.tensor(label))
    return tensor_data, tensor_labels, len(tensor_data)

# torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fun)



