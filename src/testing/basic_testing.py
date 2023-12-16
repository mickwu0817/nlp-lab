import io

import torch.cuda

from mickhouse_nlp_lab.tool import util


def try_gpu():
    print(f"{torch.backends.cudnn.enabled}")
    print(f"{torch.cuda.is_available()}")


def try_get_word_frequencey1():
    f = open('../../data/从百草园到三味书屋.txt', encoding='utf8')  # encoding 要使用和这个文件对应的编码
    txt = f.read()  # 读取全部字符
    f.close()
    # Key Function
    word_frequency = util.get_word_frequency(txt)
    key_list = list(word_frequency.keys())
    print(type(word_frequency))
    print(word_frequency)
    print(word_frequency.keys())
    print(len(word_frequency.keys()))
    print(key_list[653])
    print(word_frequency['是'])
    # for item in word_frequency.keys():
    #     print(f"{item} : {word_frequency[item]}")


def try_get_word_frequencey2():
    # 定义两个list分别存放两个板块的帖子数据
    academy_titles = []
    job_titles = []
    with open('../../data/academy_titles.txt', encoding='utf8') as f:
        for l in f:  # 按行读取文件
            academy_titles.append(l.strip())  # strip 方法用于去掉行尾空格
    with open('../../data/job_titles.txt', encoding='utf8') as f:
        for l in f:  # 按行读取文件
            job_titles.append(l.strip())  # strip 方法用于去掉行尾空格
    academy_text = ''.join(academy_titles)
    job_text = ''.join(job_titles)
    all_text = academy_text + job_text
    # Key Function
    word_frequency = util.get_word_frequency(all_text)
    print(len(word_frequency.keys()))


def try_get_vocabulary_frequencey():
    list1 = ['兼职', '心理学', '公众号', '寻兼', '职写手']
    list2 = ['兼职', '城市', '地理', '规划', '研究', '助理']
    all_list = list1 + list2

    vocabulary_list = util.get_vocabulary_frequency(all_list)
    key_list = list(vocabulary_list.keys())

    print(vocabulary_list.keys())
    print(len(vocabulary_list.keys()))
    print(key_list[0])
    print(key_list)
    print(util.get_index(key_list, '兼职'))
    print(util.get_index(key_list, '是'))
    for item in vocabulary_list.keys():
        print(f"{item} : {vocabulary_list[item]}")


def try_stringio():
    sio = io.StringIO()
    sio.write('hello ')
    print(f"{sio.getvalue()}")
    sio.write('word ')
    print(f"{sio.getvalue()}")
    sio.close()


if __name__ == '__main__':
    # try_gpu()
    # try_get_word_frequencey1()
    # try_get_word_frequencey2()
    # try_get_vocabulary_frequencey()
    # try_stringio()

    text = '大家好ABC'
    for i, word in enumerate(text):
        print(f"{i}:{word}")


    pass



