import collections
from collections import Counter  # Counter 可用来统计可迭代对象中，元素的数量


def get_word_frequency(text: str) -> collections.Counter:
    """ 得到每个字符的出现次数 """
    word_frequency = Counter(text)
    return word_frequency


