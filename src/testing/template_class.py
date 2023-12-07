import torch
from torch.nn import Module

"""
* Model : torch.nn.Module
* Focus on forward
  - Model 3 Status : Structure, Code, and Math
"""


class TemplateModel(Module):
    """ Model Template"""
    def __int__(self):
        super(TemplateModel, self).__init__()
        # TODO

    def forward(self):
        # TODO
        pass


class TemplateRNN(Module):
    """ RNN Template"""
    def __int__(self):
        super(TemplateRNN, self).__init__()
        # TODO

    def forward(self, input_tensor, hidden):
        word_vector = self.embedding(input_tensor)
        combined = torch.cat((word_vector, hidden), 1)
        hidden = self.input_to_hidden(combined)
        output = self.input_to_output(combined)
        return output, hidden

####

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



