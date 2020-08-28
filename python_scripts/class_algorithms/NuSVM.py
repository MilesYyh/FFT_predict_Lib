#modules import
from sklearn.svm import NuSVC
from class_algorithms import responseTraining

class NuSVM(object):

    #building
    def __init__ (self,dataset,target,kernel, nu, degree, gamma, validation):

        self.dataset=dataset
        self.target=target
        self.kernel=kernel
        self.validation=validation
        self.nu = nu
        self.degree = degree
        self.gamma = gamma

    def trainingMethod(self):

        self.model=NuSVC(kernel=self.kernel, degree=self.degree, gamma=self.gamma, nu=self.nu, probability=True)
        self.NuSVMAlgorithm=self.model.fit(self.dataset,self.target)
