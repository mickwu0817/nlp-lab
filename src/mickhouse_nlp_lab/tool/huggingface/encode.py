from transformers import BertTokenizer, BatchEncoding


def get_bert_tokenizer() -> BertTokenizer:
    # /home/mick/path/dev/python/mick/nlp-lab/resource/hugging_face
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path='bert-base-chinese',
        cache_dir='/home/mick/path/dev/python/mick/nlp-lab/resource/hugging_face',  # local Location (download)
        force_download=False,
    )
    print(f'{type(tokenizer)}')
    return tokenizer


def get_basic_encode(tokenizer: BertTokenizer, sentence1: str, sentence2: str) -> list[int]:
    # 基本的编码函数
    output = tokenizer.encode(
        text=sentence1,
        text_pair=sentence2,
        truncation=True, # 当句子长度大于max_length时截断
        padding='max_length', # 一律补pad到max_length长度
        add_special_tokens=True,
        max_length=25,
        return_tensors=None,  # 可取值tf(TensorFlow),pt(PyTorch),np(NumPy),默认为返回list
    )
    return output


def get_batch_encode(tokenizer: BertTokenizer, pairs: list[list[str]]) -> BatchEncoding:
    """
    :param pairs: 批量编码成对的句子

    input_ids: 编码后的词

    token_type_ids: 第二个句子的位置是1, 其他句子(第一个句子和特殊符号) 的位置是0

    special_tokens_mask: 特殊符号的位置是1,其他位置是0

    attention_mask: pad的位置是0,其他位置是1

    length: 句子长度
    """
    output = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=pairs,  # 编码成对的句子
        add_special_tokens=True,
        truncation=True,  # 当句子长度大于max_length时截断
        padding='max_length',  # 一律补零到max_length长度
        max_length=25,

        return_tensors=None,  # 可取值tf(TensorFlow),pt(PyTorch),np(NumPy),默认为返回list
        return_token_type_ids=True,  # 返回 token_type_ids 编码后的词
        return_attention_mask=True,  # 返回 attention_mask 第二个句子的位置是1, 其他句子(第一个句子和特殊符号) 的位置是0
        return_special_tokens_mask=True,  # 返回 special_tokens_mask 特殊符号的位置是1,其他位置是0
        # return_offsets_mapping=True,  # 返回 offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
        return_length=True,  # 返回 length 句子长度
    )
    return output


if __name__ == '__main__':
    tokenizer = get_bert_tokenizer()

    # Test: 添加新词
    vocab = tokenizer.get_vocab()  # 获取字典
    print(f"{type(vocab)}, {len(vocab)}, {'风景' in vocab}")
    tokenizer.add_tokens(new_tokens=['风景', '装饰', '窗子'])  # 添加新词
    tokenizer.add_special_tokens({'eos_token': '[EOS]'})  # 添加特別符號
    vocab = tokenizer.get_vocab()
    print(f"{type(vocab)}, {len(vocab)}, {'风景' in vocab}")

    # 准备实验数据
    sents = ['你站在桥上看风景', '看风景的人在楼上看你', '明月装饰了你的窗子', '你装饰了别人的梦',]

    # Test : 基本的编码函数
    basic_encode = get_basic_encode(tokenizer, sents[0], sents[1])
    print(type(basic_encode))
    print(tokenizer.decode(basic_encode))

    # Test :  批次的编码函数
    batch_encode = get_batch_encode(tokenizer, [[sents[0], sents[1]], [sents[2], sents[3]]])
    for k, v in batch_encode.items():
        print(k, ':', v)
    tokenizer.decode(batch_encode['input_ids'][0])
    pass


