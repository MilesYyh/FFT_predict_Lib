from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

class RandomForest(object):
    def __init__(self, dataset,response,n_estimators,criterion, min_samples_split, min_samples_leaf, bootstrap):
        self.dataset=dataset
        self.response=response
        self.n_estimators=n_estimators
        self.criterion=criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap

    def trainingMethod(self):
        self.model=RandomForestRegressor(n_estimators=self.n_estimators,criterion=self.criterion, min_samples_leaf=self.min_samples_leaf, min_samples_split=self.min_samples_split, bootstrap=self.bootstrap, n_jobs=-1)
        self.model=self.model.fit(self.dataset,self.response)
        cross_val_score(self.model, self.dataset, self.response, cv=10)
