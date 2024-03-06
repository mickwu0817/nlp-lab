from opencc import OpenCC
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

classifier = pipeline("sentiment-analysis")
question_answerer = pipeline("question-answering")
translator = pipeline("translation_en_to_de")

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
en2zh_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
en2zh_translator = pipeline(task="translation_en_to_zh", model=en2zh_model, tokenizer=tokenizer)


def transfer_en2zh(sentence: str) -> list[dict]:
    result = en2zh_translator(sentence, max_length=20)
    return result


def get_sentence_emotion(en_sentence: str) -> dict:
    result = classifier(en_sentence)[0]
    return result


def get_question_answer(en_context: str, en_question: str) -> dict:
    result = question_answerer(question=en_question, context=context,)
    return result


def transfer_en2de(sentence: str) -> list[dict]:
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

    cc = OpenCC('s2t')
    print(f"{transfer_en2zh('What is extractive question answering')}")
    print(f"{cc.convert(transfer_en2zh('What is extractive question answering')[0]['translation_text'])}")









