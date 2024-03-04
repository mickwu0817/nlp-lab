from datasets import load_dataset, load_from_disk, load_dataset_builder


def f1(data):
    return data['text'].startswith('非常不错')


def f2(data):
    text = data['text']
    text = ['My sentence: ' + i for i in text]
    data['text'] = text
    return data


if __name__ == '__main__':
    # /home/mick/path/dev/python/mick/nlp-lab/resource/hugging_face
    huggingface_data_path = "seamew/ChnSentiCorp"
    local_data_path = "/home/mick/path/dev/python/mick/nlp-lab/resource/hugging_face/data/ChnSentiCorp"
    output_path = "/home/mick/path/dev/python/mick/nlp-lab/resource/hugging_face/output"

    # dataset = load_dataset(path=huggingface_data_path)  # Load from Hugging Face
    # dataset.save_to_disk(dataset_dict_path=local_data_path)  # Save on Local
    dataset = load_from_disk(local_data_path)  # Load from Local
    print(f"{dataset}")
    print(f"{dataset['test'][0]}")
    print(f"{type(dataset)}")

    print(f"{dataset.remove_columns(['text'])}")  # 删除字段
    print(f"{dataset.rename_column('text', 'text_rename')}")  # 字段重命名
    dataset.set_format(type='torch', columns=['label'], output_all_columns=True)  # 设置数据格式
    print(f"{dataset['test'][0]}")

    print(f"{dataset.filter(f1)}")  # 过滤数据
    print(f"{dataset.map(function=f2, batched=True, batch_size=1000, num_proc=4)}")  # 資料前處理


    pass
