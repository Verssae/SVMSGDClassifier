import argparse
import os
import struct
import pandas as pd
import numpy as np
from SVMSGDClassifier import SVMSGDClassifier
from sklearn import metrics
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

class BinarySVMSGDClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1, eta=0.01, batch_size=1, max_epoch=1000, random_state=1):
        self.C = C
        self.eta = eta
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.random_state = random_state

    def fit(self, X, y=None):
        # y = np.where(y==0, -1, 1)
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
        print(X.shape[0])
        return ( (np.count_nonzero(y == self.predict(X))) / X.shape[0] )

class SVMSGDClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1, eta=0.001, batch_size=1, max_epoch=1000, random_state=1):
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
            self.clfs_per_classs_.append(BinarySVMSGDClassifier(self.C, self.eta,self.batch_size,self.max_epoch,self.random_state).fit(X,Y))
        return self

    def predict(self, X):
        hyper_planes = []
        for i in range(len(self.classes_)):
            hyper_planes.append(self.clfs_per_classs_[i].decision_function(X))

        hyper_planes = np.array(hyper_planes)
        n_samples = X.shape[0]
        predictions = []
        for sample in range(n_samples):
            p = hyper_planes[:,sample]
            idx = np.argmax(p)
            predictions.append(self.classes_[idx])
        return predictions

def read(dataset = "D1", path = "."):

    if dataset == "D1":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "D2":
        fname_img = os.path.join(path, 'test-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'test-labels.idx1-ubyte')
    else:
        raise Exception("dataset")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

def read_only_img(path="./dataset"):
    fname_img = os.path.join(path, 'testall-images.idx3-ubyte')
    
    lbl = []
    for i in range(0, 60000): # initialize labels with 10
        lbl.append(10)
    
    lbl = np.array(lbl)
    
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    
    get_img = lambda idx: (lbl[idx], img[idx])
    
    for i in range(len(lbl)):
        yield get_img(i)
def main():
    parser = argparse.ArgumentParser(
        description="Team1's MNIST classifier using SVMSGDClassifier")
    parser.add_argument('training_data', type=str,
                        help="Put Training Dataset's name(D1|D2)")
    parser.add_argument('test_data', type=str,
                        help="Put Test Dataset's name(D1|D2|D3)")

    args = parser.parse_args()
    
    tr = args.training_data
    tt = args.test_data
    train = list(read(tr,"./dataset"))
    
    X=[]
    for i in range(np.array(train).shape[0]):
        X.append(np.ravel(train[i][1]))
    Y=[]
    for i in range(np.array(train).shape[0]):
        Y.append(train[i][0])
    X = np.array(X)
    Y = np.array(Y)

    if (tt == "D3"):
        test = list(read_only_img())
        X_test=[]
        for i in range(np.array(test).shape[0]):
            X_test.append(np.ravel(test[i][1]))
        X_test = np.array(X_test)
        clf  = SVMSGDClassifier(C=100, max_epoch=1000, batch_size=32)
        Y_pred = clf.fit(X,Y).predict(X_test)
        file = open('./prediction.txt','w')
        for i in Y_pred:
            file.write('%d\n' %i) 
        file.close()
    else:
        test = list(read(tt,"./dataset"))
        X_test=[]
        for i in range(np.array(test).shape[0]):
            X_test.append(np.ravel(test[i][1]))
        Y_true=[]
        for i in range(np.array(test).shape[0]):
            Y_true.append(np.ravel(test[i][0]))
        X_test = np.array(X_test)
        Y_true = np.array(Y_true)
        clf  = SVMSGDClassifier(C=100, max_epoch=1000, batch_size=32)
        Y_pred = clf.fit(X,Y).predict(X_test)
        print(metrics.f1_score(Y_true,Y_pred,average='macro'))
    

    

if __name__ == '__main__':
    main()
