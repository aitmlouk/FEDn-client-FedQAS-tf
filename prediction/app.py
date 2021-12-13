from flask import Flask, render_template, request, json, jsonify, send_file
from models.squad_model import create_seed_model
from process_data import create_squad_examples, create_inputs_targets
from io import BytesIO
import numpy as np
from bson.json_util import dumps
from storage.mongotracer import MongoTracer
from storage.miniorepo import MinioRepo
import yaml
from datetime import datetime
import uuid

app = Flask(__name__)


def load_model(path):
    a = np.load(path)
    weights = []
    for i in range(len(a.files)):
        weights.append(a[str(i)])
    return weights


with open('settings.yaml', 'r') as fh:
    try:
        settings = dict(yaml.safe_load(fh))
    except yaml.YAMLError as e:
        raise e


@app.route('/')
def predict():
    try:
        minio = MinioRepo(settings)
        global_model_list = minio.get_global_model_list(settings['bucket'])
        return render_template('predict.html', global_model_list=global_model_list)
    except:
        raise Exception("Ops, Could not connect to minio, make sure to run FEDn base services!")


@app.route('/testing')
def testing():
    try:
        minio = MinioRepo(settings)
        global_model_list = minio.get_global_model_list(settings['bucket'])
        question_list = [
            'What project put the first Americans into space?',
            'What program was created to carry out these projects and missions?',
            'What year did the first manned Apollo flight occur?',
            'What President is credited with the original notion of putting Americans in space?',
            'Who did the U.S. collaborate with on an Earth orbit mission in 1975?',
            'How long did Project Apollo run?',
            'What program helped develop space travel techniques that Project Apollo used?',
            'What space station supported three manned missions in 1973-1974?'
        ]

        paragraph = "The Apollo program, also known as Project Apollo, was the third United States human spaceflight program " \
                    "carried out by the National Aeronautics and Space Administration(NASA), which accomplished landing the " \
                    "first humans on the Moon from 1969 to 1972. First conceived during Dwight D. Eisenhower's administration " \
                    "as a three-man spacecraft to follow the one-man Project Mercury which put the first Americans in space, " \
                    "Apollo was later dedicated to President John F. Kennedy's national goal of landing a man on the Moon and " \
                    "returning him safely to the Earth by the end of the 1960s, which he proposed in a May 25, 1961, " \
                    "address to Congress. Project Mercury was followed by the two-man Project Gemini. The first manned flight " \
                    "of Apollo was in 1968. Apollo ran from 1961 to 1972, and was supported by the two man Gemini program " \
                    "which ran concurrently with it from 1962 to 1966. Gemini missions developed some of the space travel " \
                    "techniques that were necessary for the success of the Apollo missions. Apollo used Saturn family rockets " \
                    "as launch vehicles. Apollo/Saturn vehicles were also used for an Apollo Applications Program, " \
                    "which consisted of Skylab, a space station that supported three manned missions in 1973-74, " \
                    "and the Apollo-Soyuz Test Project, a joint Earth orbit mission with the Soviet Union in 1975. "

        return render_template('testing.html', global_model_list=global_model_list,
                               question_list=question_list, paragraph=paragraph)
    except:
        raise Exception("Ops, Could not connect to minio, make sure to run FEDn base services!")


@app.route('/squad_predict', methods=['POST'])
def squad_predict():
    if request.method == 'POST':
        # upload seed file
        seed_model = request.form.get('model_type', 'nlp')
        global_model = request.form.get('global_model', '')
        context_input = request.form.get('context_input', '')
        question_input = request.form.get('question_input', '')
        minio = MinioRepo(settings)
        model = minio.get_global_model(global_model, settings['bucket'])

        # model = '1b06d401-5b60-4ff8-912b-88d825b4c91f'
        model = BytesIO(model)
        weights = load_model(model)
        model = create_seed_model()
        model.set_weights(weights)
        data = {"data":
            [
                {"title": "Context Title",
                 "paragraphs": [
                     {
                         "context": context_input,
                         "qas": [
                             {"question": question_input,
                              "id": "Q1"
                              }
                         ]}]}]}

        test_samples = create_squad_examples(data)
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
            print("QUESTION: " + test_sample.question)
            print("ANSWER: " + pred_ans)

            # Store predictions in MongoDB for feedback
            mongptrace = MongoTracer(settings['mongo_host'], settings['network_id'])
            mongptrace.set_prediction_info(global_model, context_input, question_input, pred_ans, datetime.now())

        return pred_ans

    else:
        pass


@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/global-models")
def global_models():
    try:
        minio = MinioRepo(settings)
        global_model_list = minio.get_object_list(settings['bucket'])
        return render_template('global_models.html', global_model_list=global_model_list)
    except:
        raise Exception("Ops, Could not connect to minio, make sure to run FEDn base services!")


@app.route('/active-learning')
def active_learning():
    return render_template('annotator.html')


document = {
    'title': 'title here',
    'paragraphs': [],
}


@app.route('/annotator', methods=['GET', 'POST'])
def annotator():
    if request.method == "POST":
        upcoming_data = request.get_json()
        paragraph = upcoming_data['paragraph']
        question = upcoming_data['question']
        answer = upcoming_data['answer']
        answer_start = upcoming_data['answer_start']
        paragraph_id = upcoming_data['paragraph_id']
        question_id = upcoming_data['question_id']

        if paragraph_id == '':
            paragraph_id = str(uuid.uuid1())
        if question_id == '':
            question_id = str(uuid.uuid1())

        data = {
            'works': True,
            'paragraph_id': paragraph_id,
            'question_id': question_id,
        }

        for rec in document.get('paragraphs'):
            if paragraph_id == rec.get('id'):
                for ques in rec.get('qas'):
                    if question_id == ques.get('id'):
                        ques.get('answers').append({'text': answer, 'answer_start': answer_start})
                        return jsonify(dumps(data))
                added_question = {
                    'id': question_id,
                    'question': question,
                    'answers': [
                        {
                            'text': answer,
                            'answer_start': answer_start
                        },
                    ],
                }
                rec.get('qas').append(added_question)
                return jsonify(dumps(data))
        added_paragraph = {
            'id': paragraph_id,
            'context': paragraph,
            'qas': [
                {
                    'id': question_id,
                    'question': question,
                    'answers': [
                        {
                            'text': answer,
                            'answer_start': answer_start
                        },
                    ],
                },
            ],
        }
        document.get('paragraphs').append(added_paragraph)
    return jsonify(dumps(data))


@app.route('/generate_file', methods=['GET', 'POST'])
def generate_data_file():
    json_data = {}
    json_data['data'] = [document]
    with open('data.json', 'w') as convert_file:
        convert_file.write(json.dumps(json_data))
    path = 'data.json'
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
