import collections
from collections import Counter  # Counter 可用来统计可迭代对象中，元素的数量

import torch


def get_word_frequency(text: str) -> collections.Counter:
    """ 取得每个字的出现次数 """
    word_frequency = Counter(text)
    return word_frequency


def get_vocabulary_frequency(vacabulary_list: list[str]) -> collections.Counter:
    """ 取得每个詞彙的出现次数 """
    word_frequency = Counter(vacabulary_list)
    return word_frequency


def get_index(token_list: list[str], token: str) -> int:
    """
    取得 Token List 中, Token 的 Index
    index = -2 結束 Token
    index = -1 未知 Token
    """
    if token in token_list:
        return token_list.index(token)
    else:
        return -1


def transfer_sentence_to_index_list(sentence, token_list: list[str]) -> torch.Tensor:
    index_list = torch.zeros(len(sentence), dtype=torch.long)
    for i, word in enumerate(sentence):
        index = get_index(token_list, word)
        index_list[i] = index
    return index_list


