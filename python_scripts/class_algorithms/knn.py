from sklearn.neighbors import KNeighborsClassifier
from class_algorithms import responseTraining

class knn(object):

    #building class...
    def __init__(self, dataset, response, n_neighbors, algorithm, metric,  weights, validation):

        #init attributes values...
        self.dataset = dataset
        self.response = response
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.weights = weights
        self.validation = validation

    #instance training...
    def trainingMethod(self):

        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, algorithm=self.algorithm, metric=self.metric, n_jobs=-1)#instancia
        self.knnAlgorithm = self.model.fit(self.dataset, self.response)