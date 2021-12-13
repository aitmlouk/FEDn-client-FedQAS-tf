import sys
import re
import string
import collections
import numpy as np
import tensorflow as tf
import json
import yaml
from read_data import read_data
from fedn.utils.kerashelper import KerasHelper
from models.squad_model import create_seed_model
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s)))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def normalize_text(text):
    text = text.lower()
    # Remove punctuations
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    # Remove articles
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    text = re.sub(regex, " ", text)
    # Remove extra white space
    text = " ".join(text.split())
    return text


def validate(model, data, settings):
    """
    Helper function to train the model
    :model: model to validate
    :data: validation data
    :settings: validation parameters
    :return: metrics report
    """
    print("-- RUNNING VALIDATION --", flush=True)
    x_eval, y_eval, eval_squad_examples = read_data(data, settings)
    print(f"{len(eval_squad_examples)} testing points created.")

    pred_start, pred_end = model.predict(x_eval)
    count = 0
    f1_metric = []
    eval_examples_no_skip = [_ for _ in eval_squad_examples if _.skip == False]
    for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
        squad_eg = eval_examples_no_skip[idx]
        offsets = squad_eg.context_token_to_char
        start = np.argmax(start)
        end = np.argmax(end)
        if start >= len(offsets):
            continue
        pred_char_start = offsets[start][0]
        if end < len(offsets):
            pred_char_end = offsets[end][1]
            pred_ans = squad_eg.context[pred_char_start:pred_char_end]
        else:
            pred_ans = squad_eg.context[pred_char_start:]
        normalized_pred_ans = normalize_text(pred_ans)
        normalized_true_ans = [normalize_text(_) for _ in squad_eg.all_answers]

        f1 = compute_f1(normalized_true_ans, normalized_pred_ans)
        f1_metric.append(f1)
        if normalized_pred_ans in normalized_true_ans:
            count += 1

    accuracy = count / len(y_eval[0])
    f1 = sum(f1_metric) / len(y_eval[0])
    print('-----------------------------')
    print('Exact Match Score -Accuracy:', accuracy)
    print('F1 Score:', f1)

    metrics = {
        "F1": f1,
        "accuracy": accuracy
    }

    print("-- VALIDATION COMPLETE! --", flush=True)
    return metrics


if __name__ == '__main__':
    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise e

    helper = KerasHelper()
    weights = helper.load_model(sys.argv[1])
    model = create_seed_model()
    model.set_weights(weights)
    metrics = validate(model, '../data/test.json', settings)

    with open(sys.argv[2], "w") as fh:
        fh.write(json.dumps(metrics))
