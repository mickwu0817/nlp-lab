import torch
import torch.nn as nn

from tqdm import tqdm

from testing.template.evalutaition import evaluate_rnn


## RNN #################################################################################
def train_rnn_batch(rnn: nn.Module, loss_function, learning_rate: float, epochs: int, training_data, testing_data):
    loss_sum = 0
    plot_point = 100
    all_losses =[]
    for epoch in range(epochs):
        for index, (input_sentence, target) in enumerate(tqdm(training_data)):
            output, loss = train_rnn(rnn, loss_function, input_sentence, target, learning_rate)
            loss_sum += loss
            if index % plot_point == 0:
                all_losses.append(loss_sum/plot_point)
                loss_sum = 0
        correct_number = 0
        for sentence, target in tqdm(testing_data):
            result = evaluate_rnn(rnn, sentence)
            top_n, top_i = result.top_k
            if top_i.item() == target.item():
                correct_number += 1
        print(f'Accuracy {correct_number/len(testing_data)}')


def train_rnn(rnn: nn.Module, loss_function, input_sentence: torch.Tensor, target: torch.Tensor, learning_rate: float):
    """
    input_sentence : word index list (from transfer_sentence_to_index_list())
    """
    rnn.zero_grad()
    output = run_rnn2(rnn, input_sentence)
    loss = loss_function(output.unsqueeze(dim=0), target)
    loss.backward()
    # 根据梯度更新模型的参数
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
    return output, loss.item()


def run_rnn1(rnn, input_sentence):
    hidden = rnn.initHidden()
    for i in range(input_sentence.size()[0]):
        output, hidden = rnn(input_sentence[i].unsqueeze(dim=0), hidden)
    return output


def run_rnn2(rnn: nn.Module, input_sentence: torch.Tensor) -> torch.Tensor:
    output = rnn(input_sentence.unsqueeze(dim=0))
    return output

