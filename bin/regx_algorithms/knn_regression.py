# modules import
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor


class KNN_Model(object):

    # building class...
    def __init__(self, dataset, response, n_neighbors, algorithm, metric, weights):

        # init attributes values...
        self.dataset = dataset
        self.response = response
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.weights = weights

    # instance training...
    def trainingMethod(self):

        self.model = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            metric=self.metric,
            n_jobs=-1,
        )  # instancia
        self.model = self.model.fit(self.dataset, self.response)
        cross_val_score(self.model, self.dataset, self.response, cv=10)
