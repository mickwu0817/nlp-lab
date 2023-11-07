import io

import torch.cuda

from mickhouse_nlp_lab.tool import util


def try_gpu():
    print(f"{torch.backends.cudnn.enabled}")
    print(f"{torch.cuda.is_available()}")


def try_get_word_frequencey():
    f = open('../../data/从百草园到三味书屋.txt', encoding='utf8')  # encoding 要使用和这个文件对应的编码
    txt = f.read()  # 读取全部字符
    f.close()

    word_frequency = util.get_word_frequency(txt)
    print(type(word_frequency))
    print(word_frequency)
    print(word_frequency.keys())
    print(word_frequency['是'])
    for item in word_frequency.keys():
        print(f"{item} : {word_frequency[item]}")


def try_stringio():
    sio = io.StringIO()
    sio.write('hello ')
    print(f"{sio.getvalue()}")
    sio.write('word ')
    print(f"{sio.getvalue()}")
    sio.close()


if __name__ == '__main__':
    try_gpu()
    # try_get_word_frequencey()
    # try_stringio()

    pass



