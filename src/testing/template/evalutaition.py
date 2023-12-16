import torch
import torch.nn as nn


## RNN #################################################################################
def evaluate_rnn(rnn: nn.Module, input_sentence: torch.Tensor):
    """
    input_sentence : word index list (from transfer_sentence_to_index_list())
    """
    with torch.no_grad():
        output = run_rnn(rnn, input_sentence)
        return output


def run_rnn(rnn: nn.Module, input_sentence: torch.Tensor) -> torch.Tensor:
    output = rnn(input_sentence.unsqueeze(dim=0))
    return output





