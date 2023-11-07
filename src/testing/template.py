import torch
from torch.nn import Module
from torch.util.data import Dataset


class MyModel(Module):
    def __int__(self):
        super(MyModel, self).__init__()
        # TODO

    def forward(self):
        # TODO
        pass


class MyDataset(Dataset):
    def __int__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        return example.data, example.lable


def collate_fun(batch):
    tensor_data = []
    tensor_labels = []
    for item, label in batch:
        tensor_data.append(torch.tensor(item))
        tensor_labels.append(torch.tensor(label))
    return tensor_data, tensor_labels, len(tensor_data)



