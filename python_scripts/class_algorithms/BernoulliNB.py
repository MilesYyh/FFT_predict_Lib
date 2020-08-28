from sklearn.naive_bayes import BernoulliNB
from class_algorithms import responseTraining

class Bernoulli (object):

    def __init__(self,dataset,target,validation):
        self.dataset=dataset
        self.target=target
        self.validation=validation

    def trainingMethod(self):
        self.model=BernoulliNB()
        self.BernoulliNBAlgorithm=self.model.fit(self.dataset,self.target)

