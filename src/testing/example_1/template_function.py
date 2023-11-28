import torch

def run_model(rnn_model, input_tensor):
    # TODO
    pass

def templage_rnn_train(rnn_model, input_tensor, label_tensor, loss_function):
    """ Template RNN Train Function """
    rnn_model.zero_grad()
    output = run_model(rnn_model, input_tensor)
    loss = loss_function(output, label_tensor)
    loss.backward()

    for parameter in rnn_model.parameters():
        parameter.data.add_(parameter.grad.data, alpha=-learning_rate)

    return output, loss.items()


def template_evaluate(model, input_tensor):
    with torch.no_grad():
        output = run_model(model, input_tensor)
        return output


