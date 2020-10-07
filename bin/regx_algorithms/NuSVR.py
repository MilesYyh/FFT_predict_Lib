# modules import
from sklearn.svm import NuSVR
from sklearn.model_selection import cross_val_score


class NuSVRModel(object):

    # building
    def __init__(self, dataset, response, kernel, degree, gamma, nu):

        # init attributes values...
        self.dataset = dataset
        self.response = response
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.nu = nu

    # instance training...
    def trainingMethod(self):

        self.model = NuSVR(
            kernel=self.kernel, degree=self.degree, gamma=self.gamma, nu=self.nu
        )
        self.model = self.model.fit(self.dataset, self.response)
        cross_val_score(self.model, self.dataset, self.response, cv=10)
