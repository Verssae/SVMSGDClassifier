from sklearn.base import BaseEstimator, ClassifierMixin
import inspect
import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

class SVMSGDClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=0.01, eta=0.01, batch_size=1, max_epoch=1000, random_state=1):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def fit(self, X, y=None):
        # X, y = check_X_y(X, y)
        # check_classification_targets(y)

        r_gen = np.random.RandomState(self.random_state)
        self.w_ = r_gen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.b_, self.w_ = self.w_[0], self.w_[1:]

        for e in range(self.max_epoch):
            batch_index = np.random.choice([i for i in range(X.shape[1])], size=self.batch_size, replace=False)
            batch = zip(X[batch_index], y[batch_index])
            for xi, yi in batch:
                if yi * (np.dot(self.w_, xi)+self.b_) < 1:
                    gradient_w = -1 * yi * xi + (1/self.C) * self.w_
                    gradient_b = -1 * yi
                else:
                    gradient_w = (1/self.C) * self.w_
                    gradient_b = 0
                self.w_ = np.add(self.w_, -1 * self.eta * gradient_w)
                self.b_ = np.add(self.b_,-1 * self.eta * gradient_b) 
                
        return self

    def decision_function(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.sign(self.decision_function(X))

if __name__ == "__main__":
    check_estimator(SVMSGDClassifier)

    
