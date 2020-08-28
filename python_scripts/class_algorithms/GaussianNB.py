from sklearn.naive_bayes import GaussianNB
from class_algorithms import responseTraining

class Gaussian(object):

    def __init__(self,dataset,target,validation):
        self.dataset=dataset
        self.target=target
        self.validation=validation

    def trainingMethod(self):

        self.model=GaussianNB()
        self.GaussianNBAlgorithm=self.model.fit(self.dataset,self.target)
