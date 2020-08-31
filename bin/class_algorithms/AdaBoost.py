from sklearn.ensemble import AdaBoostClassifier
from class_algorithms import responseTraining


class AdaBoost(object):
    def __init__(self, dataset, target, n_estimators, algorithm, validation):
        self.dataset = dataset
        self.target = target
        self.n_estimators = n_estimators
        self.algorithm = algorithm
        self.validation = validation

    def trainingMethod(self):

        self.model = AdaBoostClassifier(
            n_estimators=self.n_estimators, algorithm=self.algorithm
        )
        self.model = self.model.fit(self.dataset, self.target)
