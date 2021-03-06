import numpy as np
import os
import yaml
from fedn.utils.kerashelper import KerasHelper
from models.squad_model import create_seed_model
from read_data import create_squad_examples, create_inputs_targets
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer


def predict(global_model):
    print("-- RUNNING PREDICTION --", flush=True)
    helper = KerasHelper()
    weights = helper.load_model(global_model)
    model = create_seed_model()
    model.set_weights(weights)

    slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    save_path = "bert_base_uncased/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    slow_tokenizer.save_pretrained(save_path)

    # Load the fast tokenizer from saved file
    tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)

    # use model
    data = {"data":
        [
            {"title": "Project Apollo",
             "paragraphs": [
                 {
                     "context": "The Apollo program, also known as Project Apollo, was the third United States human "
                                "spaceflight program carried out by the National Aeronautics and Space Administration ("
                                "NASA), which accomplished landing the first humans on the Moon from 1969 to 1972. First "
                                "conceived during Dwight D. Eisenhower's administration as a three-man spacecraft to "
                                "follow the one-man Project Mercury which put the first Americans in space, Apollo was "
                                "later dedicated to President John F. Kennedy's national goal of landing a man on the "
                                "Moon and returning him safely to the Earth by the end of the 1960s, which he proposed in "
                                "a May 25, 1961, address to Congress. Project Mercury was followed by the two-man Project "
                                "Gemini. The first manned flight of Apollo was in 1968. Apollo ran from 1961 to 1972, "
                                "and was supported by the two man Gemini program which ran concurrently with it from 1962 "
                                "to 1966. Gemini missions developed some of the space travel techniques that were "
                                "necessary for the success of the Apollo missions. Apollo used Saturn family rockets as "
                                "launch vehicles. Apollo/Saturn vehicles were also used for an Apollo Applications "
                                "Program, which consisted of Skylab, a space station that supported three manned missions "
                                "in 1973-74, and the Apollo-Soyuz Test Project, a joint Earth orbit mission with the "
                                "Soviet Union in 1975.",
                     "qas": [
                         {"question": "What project put the first Americans into space?",
                          "id": "Q1"
                          },
                         {"question": "What program was created to carry out these projects and missions?",
                          "id": "Q2"
                          },
                         {"question": "What year did the first manned Apollo flight occur?",
                          "id": "Q3"
                          },
                         {
                             "question": "What President is credited with the original notion of putting Americans in space?",
                             "id": "Q4"
                         },
                         {"question": "Who did the U.S. collaborate with on an Earth orbit mission in 1975?",
                          "id": "Q5"
                          },
                         {"question": "How long did Project Apollo run?",
                          "id": "Q6"
                          },
                         {"question": "What program helped develop space travel techniques that Project Apollo used?",
                          "id": "Q7"
                          },
                         {"question": "What space station supported three manned missions in 1973-1974?",
                          "id": "Q8"
                          }
                     ]}]}]}

    test_samples = create_squad_examples(data, tokenizer, settings)
    x_test, _ = create_inputs_targets(test_samples)
    pred_start, pred_end = model.predict(x_test)
    for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
        test_sample = test_samples[idx]
        offsets = test_sample.context_token_to_char
        start = np.argmax(start)
        end = np.argmax(end)
        pred_ans = None
        if start >= len(offsets):
            continue
        pred_char_start = offsets[start][0]
        if end < len(offsets):
            pred_ans = test_sample.context[pred_char_start:offsets[end][1]]
        else:
            pred_ans = test_sample.context[pred_char_start:]
        print("Q: " + test_sample.question)
        print("A: " + pred_ans)

    return True


if __name__ == '__main__':
    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise e

    global_model = '1b06d401-5b60-4ff8-912b-88d825b4c91f'  # your global model name from minio repo here!
    result = predict(global_model)
    print("Prediction:", result)
