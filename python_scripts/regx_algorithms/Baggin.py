from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingRegressor

class Baggin(object):

    def __init__ (self,dataset,response,n_estimators, bootstrap):
        self.dataset=dataset
        self.response=response
        self.n_estimators=n_estimators
        self.bootstrap = bootstrap

    def trainingMethod(self):
        
        self.model= BaggingRegressor(n_estimators=self.n_estimators, bootstrap=self.bootstrap, n_jobs=-1)
        self.model=self.model.fit(self.dataset,self.response)
        cross_val_score(self.model, self.dataset, self.response, cv=10)

        