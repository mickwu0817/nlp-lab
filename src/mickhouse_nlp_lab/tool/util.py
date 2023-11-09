import collections
from collections import Counter  # Counter 可用来统计可迭代对象中，元素的数量


def get_word_frequency(text: str) -> collections.Counter:
    """ 得到每个字符的出现次数 """
    word_frequency = Counter(text)
    return word_frequency


def get_vocabulary_frequency(vacabulary_list: list[str]) -> collections.Counter:
    """ 得到每个字符的出现次数 """
    word_frequency = Counter(vacabulary_list)
    return word_frequency


def get_index(token_list: list[str], token: str) -> int:
    if token in token_list:
        return token_list.index(token)
    else:
        return -1


