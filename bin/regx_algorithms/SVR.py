# modules import
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score


class SVRModel(object):

    # building
    def __init__(self, dataset, response, kernel, degree, gamma):

        # init attributes values...
        self.dataset = dataset
        self.response = response
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma

    # instance training...
    def trainingMethod(self):

        self.model = SVR(kernel=self.kernel, degree=self.degree, gamma=self.gamma)
        self.model = self.model.fit(self.dataset, self.response)
        cross_val_score(self.model, self.dataset, self.response, cv=10)
