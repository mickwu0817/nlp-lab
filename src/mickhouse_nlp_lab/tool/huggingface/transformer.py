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


class MultiHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_Q = torch.nn.Linear(32, 32)
        self.fc_K = torch.nn.Linear(32, 32)
        self.fc_V = torch.nn.Linear(32, 32)
        self.out_fc = torch.nn.Linear(32, 32)
        self.norm = torch.nn.LayerNorm(normalized_shape=32, elementwise_affine=True)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, Q, K, V, mask):
        # b句话,每句话50个词,每个词编码成32维向量
        # Q、K、V = [b, 50, 32]
        b = Q.shape[0]
        # 保留下原始的Q,后面要做短接用
        clone_Q = Q.clone()
        # 标准化
        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)
        # 线性运算,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        K = self.fc_K(K)
        V = self.fc_V(V)
        Q = self.fc_Q(Q)
        # 拆分成多个头
        # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
        # [b, 50, 32] -> [b, 4, 50, 8]
        Q = Q.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        K = K.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        V = V.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        # 计算注意力
        # [b, 4, 50, 8] -> [b, 50, 32]
        score = attention(Q, K, V, mask)
        # 计算输出,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        score = self.dropout(self.out_fc(score))
        # 短接
        score = clone_Q + score
        return score


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


def attention(Q, K, V, mask):
    # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
    # Q、K、V = [b, 4, 50, 8]
    # [b, 4, 50, 8] * [b, 4, 8, 50] -> [b, 4, 50, 50]
    # Q、K矩阵相乘,求每个词相对其他所有词的注意力
    # permute 維度換位
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))
    # 除以每个头维数的平方根,做数值缩放
    score /= 8**0.5
    # mask遮盖,mask是true的地方都被替换成-inf,这样在计算softmax的时候,-inf会被压缩到0
    # mask = [b, 1, 50, 50]
    score = score.masked_fill_(mask, -float('inf'))
    score = torch.softmax(score, dim=-1)
    # 以注意力分数乘以V,得到最终的注意力结果
    # [b, 4, 50, 50] * [b, 4, 50, 8] -> [b, 4, 50, 8]
    score = torch.matmul(score, V)
    # 每个头计算的结果合一
    # [b, 4, 50, 8] -> [b, 50, 32]
    score = score.permute(0, 2, 1, 3).reshape(-1, 50, 32)
    return score


if __name__ == '__main__':
    loader = torch.utils.data.DataLoader(dataset=Dataset(), batch_size=8, drop_last=True, shuffle=True, collate_fn=None)
    # for i, (x, y) in enumerate(loader):

    pass
