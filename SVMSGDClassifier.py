from sklearn.datasets.samples_generator import make_blobs
from sklearn.base import BaseEstimator, ClassifierMixin
import inspect
import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import scale, StandardScaler, Normalizer
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline


# class SVMSGDClassifier(BaseEstimator, ClassifierMixin):
#     def __init__(self, C=1, eta=0.001, batch_size=1, max_epoch=1000, random_state=1):
#         args, _, _, values = inspect.getargvalues(inspect.currentframe())
#         values.pop("self")
#         for arg, val in values.items():
#             setattr(self, arg, val)

#     def fit(self, X, y=None):
#         y = np.where(y==0, -1, 1)
#         r_gen = np.random.RandomState(self.random_state)
#         self.w_ = r_gen.normal(loc=0.0, scale=0.1, size=1 + X.shape[1])
#         self.b_, self.w_ = self.w_[0], self.w_[1:]
#         for e in range(self.max_epoch):
#             batch_index = np.random.choice(
#                 [i for i in range(X.shape[0])], size=self.batch_size, replace=False)
#             gradient_w = []
#             gradient_b = []
#             for xi, yi in zip(X[batch_index], y[batch_index]):
#                 if yi * (np.dot(self.w_,xi)+self.b_) < 1:
#                     gradient_w.append( -1 * yi * xi + (1/self.C) * self.w_)
#                     gradient_b.append(-1 * yi)
#                 else:
#                     gradient_w.append((1/self.C) * self.w_)
#                     gradient_b.append(0)

#             gradient_w = np.array(gradient_w)
#             gradient_b = np.array(gradient_b)
#             gradient_w = (gradient_w.sum(axis=0)) / self.batch_size
#             gradient_b = (gradient_b.sum(axis=0)) /self.batch_size
            
#             self.w_ = self.w_ -1 * self.eta * gradient_w
#             self.b_ = self.b_ -1 * self.eta * gradient_b

#         return self

#     def decision_function(self, X):

#         return np.dot(X, self.w_) + self.b_

#     def predict(self, X):
#         return np.sign(self.decision_function(X))

#     def score(self, X, y):
#         return ( (np.count_nonzero(y == self.predict(X))) / X.shape[0] )

class BinarySVMSGDClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1, eta=0.001, batch_size=1, max_epoch=1000, random_state=1):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def fit(self, X, y=None):
        y = np.where(y==0, -1, 1)
        r_gen = np.random.RandomState(self.random_state)
        self.w_ = r_gen.normal(loc=0.0, scale=0.1, size=1 + X.shape[1])
        self.b_, self.w_ = self.w_[0], self.w_[1:]
        for e in range(self.max_epoch):
            batch_index = np.random.choice(
                [i for i in range(X.shape[0])], size=self.batch_size, replace=False)
            gradient_w = []
            gradient_b = []
            for xi, yi in zip(X[batch_index], y[batch_index]):
                if yi * (np.dot(self.w_,xi)+self.b_) < 1:
                    gradient_w.append( -1 * yi * xi + (1/self.C) * self.w_)
                    gradient_b.append(-1 * yi)
                else:
                    gradient_w.append((1/self.C) * self.w_)
                    gradient_b.append(0)

            gradient_w = np.array(gradient_w)
            gradient_b = np.array(gradient_b)
            gradient_w = (gradient_w.sum(axis=0)) / self.batch_size
            gradient_b = (gradient_b.sum(axis=0)) /self.batch_size
            
            self.w_ = self.w_ -1 * self.eta * gradient_w
            self.b_ = self.b_ -1 * self.eta * gradient_b

        return self

    def decision_function(self, X):
        # print((np.dot(X, self.w_) + self.b_).shape)
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def score(self, X, y):
        return ( (np.count_nonzero(y == self.predict(X))) / X.shape[0] )

class SVMSGDClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1, eta=0.001, batch_size=1, max_epoch=1000, random_state=1):
        # args, _, _, values = inspect.getargvalues(inspect.currentframe())
        # values.pop("self")
        # for arg, val in values.items():
        #     setattr(self, arg, val)
        self.C = C
        self.eta = eta
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.random_state = random_state

    def fit(self, X, y=None):
        self.classes_ = np.unique(y)
        self.clfs_per_classs_ = []
        
        for pos in self.classes_:
            Y = np.where(y==pos, 1, -1)
            # print(Y)
            self.clfs_per_classs_.append(BinarySVMSGDClassifier(self.C, self.eta,self.batch_size,self.max_epoch,self.random_state).fit(X,Y))
        return self

    def predict(self, X):
        hyper_planes = []
        for i in range(len(self.classes_)):
            hyper_planes.append(self.clfs_per_classs_[i].decision_function(X))

        hyper_planes = np.array(hyper_planes)
        n_class, n_samples = hyper_planes.shape
        predictions = []
        for sample in range(n_samples):
            p = hyper_planes[:,sample]
            idx = np.argmax(p)
            predictions.append(self.classes_[idx])
        print(predictions)
        return predictions


    def score(self, X, y):
        print(y)
        return ( (np.count_nonzero(y == self.predict(X))) / X.shape[0] )


if __name__ == "__main__":

    X, Y = make_classification(n_classes=3,n_samples=1000, n_informative=18, n_features=20)
    clf = OneVsRestClassifier(BinarySVMSGDClassifier(max_epoch=1000, batch_size=30))
    # normal = Normalizer()
    # print(clf)
    print(clf.fit(X,Y).score(X,Y))
    print(SVMSGDClassifier(max_epoch=1000,batch_size=30).fit(X,Y).score(X, Y))
    # pipeline = Pipeline(
    #     [ ('transf', normal),('estimator', OneVsRestClassifier(BinarySVMSGDClassifier(), n_jobs=-1))])
    # pipeline.set_params(estimator__C=1)
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    # scores = cross_validate(pipeline, X, Y, cv=skf,
    #                         n_jobs=-1,  scoring=["accuracy", "f1_macro"])

    # scores = pd.DataFrame(scores)
    # print(scores)
    # scores.to_csv(f'scores_{name}.csv')

    # print(clf.score(X,Y))

    # print(SVMSGDClassifier(max_epoch=10000,batch_size=100).fit(X,Y).score(X, Y))

