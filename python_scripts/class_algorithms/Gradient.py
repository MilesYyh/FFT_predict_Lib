from sklearn.ensemble import GradientBoostingClassifier
from class_algorithms import responseTraining

class Gradient(object):

    def __init__ (self,dataset,target,n_estimators, loss, min_samples_split, min_samples_leaf, validation):
        self.dataset=dataset
        self.target=target
        self.n_estimators=n_estimators
        self.loss = loss
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.validation=validation

    def trainingMethod(self):

        self.model= GradientBoostingClassifier(n_estimators=self.n_estimators)
        self.GradientAlgorithm= self.model.fit(self.dataset,self.target)
