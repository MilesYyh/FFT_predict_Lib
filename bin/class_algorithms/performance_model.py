from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


class performance_model(object):
    def __init__(self, response_real, response_predict):
        self.response_real = response_real
        self.response_predict = response_predict

    def get_performance(self):

        self.accuracy_value = accuracy_score(self.response_real, self.response_predict)
        self.f1_value = f1_score(
            self.response_real, self.response_predict, average="weighted"
        )
        self.precision_value = precision_score(
            self.response_real, self.response_predict, average="weighted"
        )
        self.recall_value = recall_score(
            self.response_real, self.response_predict, average="weighted"
        )
