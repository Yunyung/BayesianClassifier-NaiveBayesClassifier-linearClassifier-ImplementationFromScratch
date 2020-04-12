import numpy as np
from sklearn.model_selection import KFold

class cross_validate:
    """
        Evaluate fit/score by cross-validataion

        Parameter
        ------------
        estimator: estimator object implementing 'fit'
            The object to use fit the data
        
        X : array-like, shape = [n_samples, n_features]
                Dataset, Training samples
        
        y : array-like
            The target variable to try to predict in the case of supervied learning

        cv: int 
            How many fold(split). Default 5-fold cross validation
    """


    def score(self, estimator, X, y, cv = 5):
        score_list = list()
        kf = KFold(n_splits = cv, shuffle=True)
        # Calculate each fold score
        for train_index, test_index in kf.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            #print(y[train_index])
            estimator.fit(X[train_index], y[train_index])
            score = estimator.score(X[test_index], y[test_index])
            score_list.append(score)
        return score_list





