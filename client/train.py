import sys
import yaml
import tensorflow as tf
from read_data import read_data
from fedn.utils.kerashelper import KerasHelper
from models.squad_model import create_seed_model
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def train(model, data, settings):
    """
    Helper function to train the model
    :model: model to train
    :data: training data
    :settings: training parameters
    :return: model
    """
    print("-- RUNNING TRAINING --")

    x_train, y_train, train_squad_examples = read_data(data, settings)
    print(f"{len(train_squad_examples)} training points created.")
    model.fit(x_train, y_train, epochs=settings['epochs'], batch_size=settings['batch_size'], verbose=settings['verbose'])

    print("-- TRAINING COMPLETED --")
    return model


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
    model = train(model, '../data/train.json', settings)
    helper.save_model(model.get_weights(), sys.argv[2])
