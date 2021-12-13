import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import BertTokenizer, TFBertModel, BertConfig


with open('./settings.yaml', 'r') as fh:
    try:
        settings = dict(yaml.safe_load(fh))
    except yaml.YAMLError as e:
        raise e


def create_seed_model():
    """
    Helper function to generate an initial seed model.
    Fine tune BERT model on SQuAD dataset
    :return: model
    """
    # BERT encoder
    encoder = TFBertModel.from_pretrained("bert-base-uncased")

    # QA Model
    input_ids = layers.Input(shape=(settings['max_seq_length']), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(settings['max_seq_length']), dtype=tf.int32)
    attention_mask = layers.Input(shape=(settings['max_seq_length']), dtype=tf.int32)
    embedding = encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]

    start_logits = layers.Dense(1, name="start_logit", use_bias=False)(embedding)
    start_logits = layers.Flatten()(start_logits)
    end_logits = layers.Dense(1, name="end_logit", use_bias=False)(embedding)
    end_logits = layers.Flatten()(end_logits)
    start_probs = layers.Activation(keras.activations.softmax)(start_logits)
    end_probs = layers.Activation(keras.activations.softmax)(end_logits)

    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[start_probs, end_probs],
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(learning_rate=settings['learning_rate'])
    model.compile(optimizer=optimizer, loss=[loss, loss])
    return model
