"""
Author:
mailto:
Name Classs:
Description:
Dependences:
"""

from sklearn.ensemble import BaggingClassifier
from class_algorithms import responseTraining


class Baggin(object):
    def __init__(self, dataset, target, n_estimators, bootstrap, validation):
        self.dataset = dataset
        self.target = target
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.validation = validation

    def trainingMethod(self):

        self.model = BaggingClassifier(
            n_estimators=self.n_estimators, bootstrap=self.bootstrap, n_jobs=-1
        )
        self.BagginAlgorithm = self.model.fit(self.dataset, self.target)
