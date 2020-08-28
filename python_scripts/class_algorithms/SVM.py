
#modules import
from sklearn import svm
from class_algorithms import responseTraining

class SVM(object):

    #building
    def __init__(self,dataset, target, kernel, C_value, degree, gamma, validation):

        #init attributes values...
        self.dataset=dataset
        self.target=target
        self.kernel=kernel
        self.validation=validation
        self.C_value = C_value
        self.degree = degree
        self.gamma = gamma

    #instance training...
    def trainingMethod(self):

        self.model=svm.SVC(kernel=self.kernel, degree=self.degree, gamma=self.gamma, C=self.C_value, probability=True)
        self.SVMAlgorithm =self.model.fit(self.dataset,self.target)
