import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class NeuralNetwork:
    def __init__(self, n_features, n_classes, n_neurons, n_layers, optimizer='adam', loss='sparse_categorical_crossentropy',
                 metrics=['accuracy']):
        """
        Init tensorflow neural network model
        :param n_features: number of features in data
        :param n_classes: number of data classes
        :param n_neurons: number of neurons in hidden layers
        :param n_layers: number of hidden layers
        :param optimizer: tensorflow optimizer function string
        :param loss: tensorflow loss funtion string
        :param metrics: list of tensorflow metrics
        """

        if metrics is None:
            metrics = ['accuracy']
        inputs = keras.Input(shape=(n_features,), name="features")

        assert 0 < n_layers < 3, "by now layers should be only 1 or 2"

        if n_layers == 1:
            x = layers.Dense(n_neurons, activation="relu", name="hidden_1")(inputs)

        elif n_layers == 2:
            x = layers.Dense(n_neurons, activation="relu", name="hidden_1")(inputs)
            x = layers.Dense(n_neurons, activation="relu", name="hidden_2")(x)

        outputs = layers.Dense(n_classes, activation="softmax", name="predictions")(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)

    def train_model(self, train_data, train_labels, epochs=50, batch_size=128):
        """
        Train tensorflow model
        :param train_data: examples for model fitting
        :param train_labels: labels of examples
        :param epochs: number of epochs in training
        :param batch_size: size of batches
        :return: tensorflow history object
        """

        return self.model.fit(train_data,
                              train_labels,
                              verbose=0,
                              epochs=epochs,
                              validation_split=0.2,
                              batch_size=batch_size)

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

        prediction = self.model.predict(eval_data)
        return [np.argmax(pred_probability) for pred_probability in prediction]
