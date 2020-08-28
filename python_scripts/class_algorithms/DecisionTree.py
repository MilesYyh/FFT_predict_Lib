from sklearn import tree
from class_algorithms import responseTraining

class DecisionTree(object):

    def __init__ (self, dataset, target, criterion, splitter,validation):
        self.dataset=dataset
        self.target=target
        self.criterion=criterion
        self.splitter=splitter
        self.validation=validation

    def trainingMethod(self):
        self.model=tree.DecisionTreeClassifier(criterion=self.criterion,splitter=self.splitter)
        self.DecisionTreeAlgorithm=self.model.fit(self.dataset,self.target)

