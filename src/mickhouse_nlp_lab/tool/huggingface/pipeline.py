from transformers import pipeline

classifier = pipeline("sentiment-analysis")
question_answerer = pipeline("question-answering")
translator = pipeline("translation_en_to_de")


def get_sentence_emotion(sentence: str) -> dict:
    result = classifier("I hate you")[0]
    return result


def get_question_answer(context: str, question: str) -> dict:
    result = question_answerer(question=question,context=context,)
    return result


def transfer_en2de(sentence: str) -> dict:
    result = translator(sentence, max_length=40)
    return result


if __name__ == '__main__':
    print(f"{get_sentence_emotion('I hate you')}")
    print(f"{get_sentence_emotion('I love you')}")

    context = r"""
    Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a 
    question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune 
    a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.
    """
    print(f"{get_question_answer(context, 'What is extractive question answering')}")
    print(f"{get_question_answer(context, 'What is a good example of a question answering dataset')}")

    print(f"{transfer_en2de('What is extractive question answering')}")








