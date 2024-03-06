import random
import torch

import numpy as np
from torch import Tensor


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        self.vocabulary = get_vocabulary()

    def __len__(self):
        return 100000

    def __getitem__(self, i):
        return get_data(self.vocabulary['vocab_x'], self.vocabulary['vocab_y'])


def get_vocabulary()->list[dict]:
    vocab_x = '<SOS>,<EOS>,<PAD>,0,1,2,3,4,5,6,7,8,9,q,w,e,r,t,y,u,i,o,p,a,s,d,f,g,h,j,k,l,z,x,c,v,b,n,m'
    vocab_x = {word: i for i, word in enumerate(vocab_x.split(','))}
    [k for k, v in vocab_x.items()]
    vocab_y = {k.upper(): v for k, v in vocab_x.items()}
    [k for k, v in vocab_y.items()]

    print('vocab_x=', vocab_x)
    print('vocab_y=', vocab_y)
    return {'vocab_x': vocab_x, 'vocab_y': vocab_y}


def get_data(vocab_x: dict, vocab_y: dict) -> list[Tensor]:
    # 定义词集合
    words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
             'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm']
    # 定义每个词被选中的概率
    p = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    p = p / p.sum()
    # 随机选n个词
    n = random.randint(30, 48)
    x = np.random.choice(words, size=n, replace=True, p=p)
    # 采样的结果就是x
    x = x.tolist()
    y = [f(i) for i in x]
    # 逆序
    y = y[::-1]
    # y中的首字母双写
    y = [y[0]] + y
    # 加上首尾符号
    x = ['<SOS>'] + x + ['<EOS>']
    y = ['<SOS>'] + y + ['<EOS>']
    # 补pad到固定长度
    x = x + ['<PAD>'] * 50
    y = y + ['<PAD>'] * 51
    x = x[:50]
    y = y[:51]
    # 编码成数据
    x = [vocab_x[i] for i in x]
    y = [vocab_y[i] for i in y]
    # 转tensor
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    return [x, y]


def f(i):
    # y是对x的变换得到的
    # 字母大写,数字取9以内的互补数
    i = i.upper()
    if not i.isdigit():
        return i
    i = 9 - int(i)
    return str(i)


def mask_pad(data) -> Tensor:
    vocabulary = get_vocabulary()
    vocab_x = vocabulary['vocab_x']
    # b句话,每句话50个词,这里是还没embed的
    # data = [b, 50]
    # 判断每个词是不是<PAD>
    mask = data == vocab_x['<PAD>']
    # [b, 50] -> [b, 1, 1, 50]
    mask = mask.reshape(-1, 1, 1, 50)
    # 在计算注意力时,是计算50个词和50个词相互之间的注意力,所以是个50*50的矩阵
    # 是pad的列是true,意味着任何词对pad的注意力都是0
    # 但是pad本身对其他词的注意力并不是0
    # 所以是pad的行不是true
    # 复制n次
    # [b, 1, 1, 50] -> [b, 1, 50, 50]
    mask = mask.expand(-1, 1, 50, 50)
    return mask


def mask_tril(data) -> Tensor:
    vocabulary = get_vocabulary()
    vocab_y = vocabulary['vocab_x']
    # b句话,每句话50个词,这里是还没embed的
    # data = [b, 50]
    # 50*50的矩阵表示每个词对其他词是否可见
    # 上三角矩阵,不包括对角线,意味着,对每个词而言,他只能看到他自己,和他之前的词,而看不到之后的词
    # [1, 50, 50]
    """
    [[0, 1, 1, 1, 1],
     [0, 0, 1, 1, 1],
     [0, 0, 0, 1, 1],
     [0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0]]"""
    tril = 1 - torch.tril(torch.ones(1, 50, 50, dtype=torch.long))
    # 判断y当中每个词是不是pad,如果是pad则不可见
    # [b, 50]
    mask = data == vocab_y['<PAD>']
    # 变形+转型,为了之后的计算
    # [b, 1, 50]
    mask = mask.unsqueeze(1).long()
    # mask和tril求并集
    # [b, 1, 50] + [1, 50, 50] -> [b, 50, 50]
    mask = mask + tril
    # 转布尔型
    mask = mask > 0
    # 转布尔型,增加一个维度,便于后续的计算
    mask = (mask == 1).unsqueeze(dim=1)
    return mask


if __name__ == '__main__':
    loader = torch.utils.data.DataLoader(dataset=Dataset(), batch_size=8, drop_last=True, shuffle=True, collate_fn=None)
    # for i, (x, y) in enumerate(loader):

    pass
