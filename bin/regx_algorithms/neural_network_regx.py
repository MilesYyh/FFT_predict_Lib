import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tarfile
import os.path


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


class NeuralNetwork:
    def __init__(
        self,
        n_features,
        n_neurons,
        n_layers,
        optimizer=tf.keras.optimizers.RMSprop(0.001),
        loss="mse",
        metrics=["mae", "mse"],
        activation="relu",
    ):
        """
        Init tensorflow neural network model
        :param n_features: number of features in data
        :param n_neurons: number of neurons in hidden layers
        :param n_layers: number of hidden layers
        :param optimizer: tensorflow optimizer
        :param loss: tensorflow loss funtion string
        :param metrics: list of tensorflow metrics
        """

        assert 0 < n_layers < 3, "by now layers should be only 1 or 2"

        if n_layers == 1:
            self.model = keras.Sequential(
                [
                    layers.Dense(
                        n_neurons,
                        activation=activation,
                        name="hidden_1",
                        input_shape=[n_features],
                    ),
                    layers.Dense(1, name="output"),
                ]
            )

        elif n_layers == 2:
            self.model = keras.Sequential(
                [
                    layers.Dense(
                        n_neurons,
                        activation=activation,
                        name="hidden_1",
                        input_shape=[n_features],
                    ),
                    layers.Dense(n_neurons, activation=activation, name="hidden_2"),
                    layers.Dense(1, name="output"),
                ]
            )

        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def train_model(self, train_data, train_labels, epochs=50, batch_size=128):
        """
        Train tensorflow model
        :param train_data: examples for model fitting
        :param train_labels: labels of examples
        :param epochs: number of epochs in training
        :param batch_size: size of batches
        :return: tensorflow history object
        """

        return self.model.fit(
            train_data,
            train_labels,
            verbose=0,
            epochs=epochs,
            validation_split=0.2,
            batch_size=batch_size,
        )

    def test_model(self, test_data, test_labels):
        """
        Test trained tensorflow model
        :return: dictionary with testing metrics
        """
        return self.model.evaluate(test_data, test_labels, verbose=2, return_dict=True)

    def predict(self, eval_data):
        """
        Predict values using trained model
        :param eval_data: data for evaluation with no label
        :return: predicted label for data
        """
        return self.model.predict(eval_data).flatten()

    def get_model(self):
        """
        Get tensorflow model
        :return: tf model object
        """
        return self.model

    def save_model(self, path, overwrite=True):
        """
        Save model
        :param path: path to save
        :param overwrite: overwrite file
        """

        self.model.save(path, overwrite=overwrite)
        make_tarfile(path + ".tar", path)
