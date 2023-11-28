import jieba
import pkuseg


def cut_by_jieba(sentence: str) -> list:
    result = list(jieba.cut(sentence))
    return result


def cut_by_pkuseg(sentence: str) -> list:
    segmentor = pkuseg.pkuseg(postag=True)
    result = segmentor.cut(sentence)
    return result


if __name__ == '__main__':
    sentence = '國內恐爆黴漿菌感染潮 醫揭隱憂「原廠藥缺貨1年多」食藥署回應了'
    print(f'{cut_by_jieba(sentence)}')
    print(f'{cut_by_pkuseg(sentence)}')



