from sklearn.model_selection import cross_val_score
from sklearn import tree


class DecisionTree(object):
    def __init__(self, dataset, response, criterion, splitter):
        self.dataset = dataset
        self.response = response
        self.criterion = criterion
        self.splitter = splitter

    def trainingMethod(self):
        self.model = tree.DecisionTreeRegressor(
            criterion=self.criterion, splitter=self.splitter
        )
        self.model = self.model.fit(self.dataset, self.response)
        cross_val_score(self.model, self.dataset, self.response, cv=10)
