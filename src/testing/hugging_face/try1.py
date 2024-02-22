from transformers import BertTokenizer

if __name__ == '__main__':
    # /home/mick/path/dev/python/mick/nlp-lab/resource/hugging_face
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path='bert-base-chinese',
        cache_dir='/home/mick/path/dev/python/mick/nlp-lab/resource/hugging_face',
        force_download=False,
    )
    print(f'{tokenizer}')

    pass