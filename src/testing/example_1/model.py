import torch
import torch.nn as nn


class WordRNN(nn.Module):
    def __init__(self, dict_words_size, embedding_size, hidden_size, output_size):
        super(WordRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(dict_words_size, embedding_size)
        self.i2h = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(embedding_size + hidden_size, output_size)
        # LogSoftmax其实就是对softmax的结果进行log
        # dim=0：对每一列的所有元素进行softmax运算，并使得每一列所有元素和为1
        # dim=1：对每一行的所有元素进行softmax运算，并使得每一行所有元素和为1。
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden):
        word_vector = self.embedding(input_tensor)
        print(f"{word_vector}")
        combined = torch.cat((word_vector, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
