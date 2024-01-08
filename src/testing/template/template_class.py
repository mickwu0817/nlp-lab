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
class SelfRNN(nn.Module):
    """
    Self RNN : 需自己建立 Loop 處理
    """

    def __init__(self, word_count, embedding_size, hidden_size, output_size):
        super(SelfRNN, self).__init__()

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


class PyTorchRNN(nn.Module):
    """
    PyTorch RNN : 不再需要自己使用 Loop 處理 Token, 而是可以一次傳入所有 Tokens
    """

    def __init__(self, word_count: int, embedding_size: int, hidden_size: int, output_size: int, num_layers: int):
        super(PyTorchRNN, self).__init__()
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

    def forward(self, input_tensor: torch.Tensor):
        """
        input_tensor = [input_size, batch_size] # for transfer so batch_size in dimension 1, see last is output
        embedded = [input_size, batch_size, embedding_size]
        outputs  = [input_size, batch_size, hidden_size * n directions]
        hidden   = [layer_size * n directions, batch_size, hidden_size]
        cell     = [layer_size * n directions, batch_size, hidden_size]
        outputs are always from the top hidden layer
        """
        embedded = self.dropout(self.embedding(input_tensor))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_size: int, embedding_size: int, hidden_size: int, layer_size: int, dropout_ratio: float):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, layer_size, dropout=dropout_ratio)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input_tensor: torch.Tensor, hidden, cell):
        """
        input_tensor  = [batch size]
        hidden = [layer_size * n directions, batch_size, hidden_size]
        cell   = [layer_size * n directions, batch_size, hidden_size]
        "n directions in the decoder will both always be 1", therefore:
        hidden = [layer_size, batch_size, hidden_size]
        context = [layer_size, batch_size, hidden_size]
        input_tensor = [1, batch size]
        embedded = [1, batch size, emb dim]

        output = [seqence_size, batch_size, hidden_size * n directions]
        hidden = [layer_size * n directions, batch_size, hidden_size]
        cell   = [layer_size * n directions, batch_size, hidden_size]
        "seqence_size and n directions will always be 1 in the decoder", therefore:
        output = [1, batch_size, hidden_size]
        hidden = [layer_size, batch_size, hidden_size]
        cell   = [layer_size, batch_size, hidden_size]
        prediction = [batch_size, output_size]
        """
        input_tensor = input_tensor.unsqueeze(0)
        embedded = self.dropout(self.embedding(input_tensor))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.linear(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, input_total_word_size, output_word_size, encode_embedding_size, decode_embedding_size,
                       hidden_size: int, layer_size: int, encode_dropout_ratio: float, decode_dropouts_ratio: float, device):
        super().__init__()
        self.encoder = Encoder(input_total_word_size, encode_embedding_size, hidden_size, layer_size, encode_dropout_ratio)
        self.decoder = Decoder(output_word_size, decode_embedding_size, hidden_size, layer_size, decode_dropouts_ratio)
        self.device = device

    def forward(self, sorce_tensors, target_tensors, teacher_forcing_ratio=0.5):
        """
        srsorce_tensors = [sorce_tensor_size, batch_size]
        target_tensors = [target_tensor_size, batch size]
        teacher_forcing_ratio is probability to use teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        tensor to store decoder outputs
        last hidden state of the encoder is used as the initial hidden state of the decoder
        first input to the decoder is the <sos> tokens

        In Loop
        insert input token embedding, previous hidden and previous cell states
        receive output tensor (predictions) and new hidden and cell states
        place predictions in a tensor holding predictions for each token
        decide if we are going to use teacher forcing or not
        get the highest predicted token from our predictions
        if teacher forcing, use actual next token as next input
        if not, use predicted token
        """
        batch_size = target_tensors.shape[1]
        target_tensor_size = target_tensors.shape[0]
        target_single_word_size = self.decoder.output_size
        outputs = torch.zeros(target_tensor_size, batch_size, target_single_word_size).to(self.device)

        hidden, cell = self.encoder(sorce_tensors)
        input_single_word = target_tensors[0, :]
        for t in range(1, target_tensor_size):
            output_single_word, hidden, cell = self.decoder(input_single_word, hidden, cell)
            outputs[t] = output_single_word
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output_single_word.argmax(1)
            input_single_word = target_tensors[t] if teacher_force else top1
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



