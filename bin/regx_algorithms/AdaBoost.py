from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostRegressor


class AdaBoost(object):
    def __init__(self, dataset, response, n_estimators, loss):
        self.dataset = dataset
        self.response = response
        self.n_estimators = n_estimators
        self.loss = loss

    def trainingMethod(self):
        self.model = AdaBoostRegressor(n_estimators=self.n_estimators, loss=self.loss)
        self.model = self.model.fit(self.dataset, self.response)
        cross_val_score(self.model, self.dataset, self.response, cv=10)
