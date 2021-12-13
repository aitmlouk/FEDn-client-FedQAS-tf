import json
import os
import numpy as np
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer


class SquadExample:
    """
    Process SQUAD dataset
    """

    def __init__(self, question, context, start_char_idx, answer_text, all_answers):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.skip = False
        self.start_token_idx = -1
        self.end_token_idx = -1

    def preprocess(self, tokenizer, settings):
        context = self.context
        question = self.question
        answer_text = self.answer_text
        start_char_idx = self.start_char_idx

        # Clean context, answer and question
        context = " ".join(str(context).split())
        question = " ".join(str(question).split())
        answer = " ".join(str(answer_text).split())

        # Tokenize context
        tokenized_context = tokenizer.encode(context)
        # Tokenize question
        tokenized_question = tokenizer.encode(question)

        if self.answer_text is not None:
            # Find end character index of answer in context
            end_char_idx = start_char_idx + len(answer)
            if end_char_idx >= len(context):
                self.skip = True
                return

            # Mark the character indexes in context that are in answer
            is_char_in_ans = [0] * len(context)
            for idx in range(start_char_idx, end_char_idx):
                is_char_in_ans[idx] = 1

            # Find tokens that were created from answer characters
            ans_token_idx = []
            for idx, (start, end) in enumerate(tokenized_context.offsets):
                if sum(is_char_in_ans[start:end]) > 0:
                    ans_token_idx.append(idx)

            if len(ans_token_idx) == 0:
                self.skip = True
                return

            # Find start and end token index for tokens from answer
            self.start_token_idx = ans_token_idx[0]
            self.end_token_idx = ans_token_idx[-1]
        # Create inputs
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
            tokenized_question.ids[1:]
        )
        attention_mask = [1] * len(input_ids)

        # Pad and create attention masks.
        # Skip if truncation is needed
        padding_length = settings['max_seq_length'] - len(input_ids)
        if padding_length > 0:  # pad
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:  # skip
            self.skip = True
            return

        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.context_token_to_char = tokenized_context.offsets


def create_squad_examples(raw_data, tokenizer, settings):
    squad_examples = []
    for item in raw_data["data"]:
        for para in item["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                question = qa["question"]
                if "answers" in qa:
                    answer_text = qa["answers"][0]["text"]
                    all_answers = [_["text"] for _ in qa["answers"]]
                    start_char_idx = qa["answers"][0]["answer_start"]
                    squad_eg = SquadExample(question, context, start_char_idx, answer_text, all_answers)
                else:
                    squad_eg = SquadExample(question, context, start_char_idx=None, answer_text=None, all_answers=None)
                squad_eg.preprocess(tokenizer, settings)
                squad_examples.append(squad_eg)
    return squad_examples


def create_inputs_targets(squad_examples):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in squad_examples:
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y


def read_data(filename, settings):
    """
    Helper function to read and preprocess SQUAD data for training and validation with Keras.
    :return: test, training data or validation data and nbr of examples
    """
    slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    save_path = "bert_base_uncased/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    slow_tokenizer.save_pretrained(save_path)

    # Load the fast tokenizer from saved file
    tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)
    with open(filename) as f: raw_train_data = json.load(f)
    train_squad_examples = create_squad_examples(raw_train_data, tokenizer, settings)
    x_train, y_train = create_inputs_targets(train_squad_examples)

    return x_train, y_train, train_squad_examples
