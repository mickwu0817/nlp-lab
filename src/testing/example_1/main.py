import json
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

from tqdm import tqdm
from testing.example_1 import model


def train():
    ''' https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html '''
    # Preprocessing #
    academy_path = './src/testing/example_1/data/academy_titles.txt'
    academy_linelist = transfer_text_to_linelist(academy_path)
    job_path = './src/testing/example_1/data/job_titles.txt'
    job_linelist = transfer_text_to_linelist(job_path)
    wordlist_dict = get_wordlist_dict(academy_linelist, job_linelist)
    # Training
    epoch = 1
    learning_rate = 0.005
    embedding_size = 200
    hidden_layer_size = 10
    category_size = 2
    split_ratio = 0.7
    loss_function = torch.nn.NLLLoss()
    rnn = model.WordRNN(len(wordlist_dict) + 1, embedding_size, hidden_layer_size, category_size)
    train_data, test_data = split_data(get_all_data_with_tag(academy_linelist, job_linelist, wordlist_dict), split_ratio)
    result = get_training_loss_and_train(train_data, test_data, rnn, epoch, learning_rate)
    # Save & Show Result
    save_dict_list(wordlist_dict, './src/testing/example_1/dict/word_dict.json')
    torch.save(rnn, './src/testing/example_1/model/rnn_model.pkl')
    show_loss(result)


def transfer_text_to_linelist(path: str) -> list:
    line_list = []
    with open(path, encoding='utf8') as file:
        for line in file:  # 按行读取文件
            line_list.append(line.strip())  # strip 方法用于去掉行尾空格
    return line_list


def get_wordlist_dict(academy_titles: list, job_titles: list) -> list:
    word_set = set()
    for title in academy_titles:
        for word in title:
            word_set.add(word)
    for title in job_titles:
        for word in title:
            word_set.add(word)

    word_list = list(word_set)
    # save word list
    # with open('word_list', 'w') as f:
    #     json.dump(word_list, f)
    return word_list


def transfer_line_to_tensor_by_wordlist_dict(line: list, wordlist: list):
    wordlist_size = len(wordlist) + 1  # 加一个 unknown word
    tensor = torch.zeros(len(line), dtype=torch.long)
    for line_index, ch in enumerate(line):
        try:
            wordlist_index = wordlist.index(ch)
        except ValueError:
            wordlist_index = wordlist_size - 1
        tensor[line_index] = wordlist_index
    return tensor


def describe_input_tensor(input_tensor: torch.Tensor):
    print(f'input_tensor: {input_tensor}')
    print(f'{input_tensor.size()[0]}')
    print(f'{input_tensor[0]} {type(input_tensor[0])}')
    print(f'{input_tensor[0].unsqueeze(dim=0)}')


def get_all_data_with_tag(academy_linelist: list, job_linelist: list, wordlist_dict: list) -> list:
    # categories = ["考研考博", "招聘信息"]
    all_data = []
    for line in academy_linelist:
        all_data.append((transfer_line_to_tensor_by_wordlist_dict(line, wordlist_dict), torch.tensor([0], dtype=torch.long)))
    for line in job_linelist:
        all_data.append((transfer_line_to_tensor_by_wordlist_dict(line, wordlist_dict), torch.tensor([1], dtype=torch.long)))
    return all_data


def split_data(all_data: list, split_ratio):
    random.shuffle(all_data)
    data_len = len(all_data)
    train_data = all_data[:int(data_len * split_ratio)]
    test_data = all_data[int(data_len * split_ratio):]
    print("Train data size: ", len(train_data))
    print("Test data size: ", len(test_data))
    return train_data, test_data


def get_training_loss_and_train(train_data, test_data, rnn_model, epoch, learning_rate):
    loss_function = torch.nn.NLLLoss()
    loss_sum = 0
    all_losses = []
    plot_every = 100
    for e in range(epoch):
        for ind, (title_tensor, label) in enumerate(tqdm(train_data)):
            output, loss = train_one_time(rnn_model, loss_function, title_tensor, label, learning_rate)
            loss_sum += loss
            if ind % plot_every == 0:
                all_losses.append(loss_sum / plot_every)
                loss_sum = 0
        c = 0
        for title, category in tqdm(test_data):
            output = evaluate(rnn_model, title)
            topn, topi = output.topk(1)
            if topi.item() == category[0].item():
                c += 1
        print('accuracy', c / len(test_data))
    return all_losses


def train_one_time(rnn_model, loss_function, input_tensor, category_tensor, learning_rate):
    rnn_model.zero_grad()
    output = run_one_line_rnn(rnn_model, input_tensor)
    loss = loss_function(output, category_tensor)
    loss.backward()
    # 根据梯度更新模型的参数
    for p in rnn_model.parameters():
        try:
            # y = y + alpha * x
            p.data.add_(p.grad.data, alpha=-learning_rate)
        except Exception as e:
            print(f"\n{e}")
    return output, loss.item()


def run_one_line_rnn(rnn_model, input_tensor):
    hidden = rnn_model.initHidden()
    for i in range(input_tensor.size()[0]):
        output, hidden = rnn_model(input_tensor[i].unsqueeze(dim=0), hidden)
    return output


def show_loss(loss: list):
    plt.figure(figsize=(10, 7))
    plt.ylabel('Average Loss')
    plt.plot(loss[1:])
    plt.show()


def save_dict_list(dict_list: list, path: str):
    with open(path, 'w') as file:
        json.dump(dict_list, file)


def load_dict_list(path: str) -> list:
    with open(path, 'r') as file:
        dict_list = json.load(file)
        return dict_list


def test(line: str):
    # line = '我要考博班'
    categories = ['考研考博', '招聘信息']
    rnn_model = torch.load('./src/testing/example_1/model/rnn_model.pkl')
    dict_list = load_dict_list('./src/testing/example_1/dict/word_dict.json')
    result = get_category(rnn_model, categories, dict_list, line)
    print(f"{result}")
    return result


def get_category(rnn_model, categories, dict_list, line):
    input_tensor = transfer_line_to_tensor_by_wordlist_dict(line, dict_list)
    print(f"{input_tensor}")
    output = evaluate(rnn_model, input_tensor)
    topn, topi = output.topk(1)
    return categories[topi.item()]


def evaluate(rnn_model, input_tensor):
    with torch.no_grad():
        # hidden = rnn_model.initHidden()
        output = run_one_line_rnn(rnn_model, input_tensor)
        return output


if __name__ == '__main__':
    test_sentence = '我要考博班'
    train()
    test(test_sentence)

