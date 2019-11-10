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

def main():
    # parser = argparse.ArgumentParser(
    #     description="Team1's MNIST classifier using SVMSGDClassifier")
    # parser.add_argument('training_data', type=str,
    #                     help="Put Training Dataset's name(D1|D2)")
    # parser.add_argument('test_data', type=str,
    #                     help="Put Test Dataset's name(D1|D2|D3)")

    # args = parser.parse_args()
    # tr = args.training_data
    # tt = args.test_data
    # tr_x, tr_x_lbl = load_dataset(tr)
    # tt_x, tt_x_lbl = load_dataset(tt)
    train = list(read("training","./dataset"))
    X=[]
    for i in range(60000):
        X.append(np.ravel(train[i][1]))
    Y=[]
    for i in range(60000):
        Y.append(train[i][0])
    test = list(read("testing","./dataset"))
    X_test=[]
    for i in range(10000):
        X_test.append(np.ravel(test[i][1]))
    Y_true=[]
    for i in range(10000):
        Y_true.append(np.ravel(test[i][0]))
    X = np.array(X)
    Y = np.array(Y)
    X_test = np.array(X_test)
    Y_true = np.array(Y_true)
    normal = Normalizer()
    scalar = StandardScaler()
    clf  = SVMSGDClassifier(max_epoch=1000,batch_size=32)
    pipeline = Pipeline(
        [ ('transformer',scalar),('estimator', clf)])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    parameters = {'estimator__C': [ 0.1, 1, 10, 100, 1000, 10000]}

    print("[Finding hyparmas]")

    clf = GridSearchCV(pipeline, param_grid=parameters,
                       n_jobs=-1, cv=skf, verbose=True)
    # clf.fit(X, Y)
    clf.fit(X,Y)
    cv_results = pd.DataFrame(clf.cv_results_)
    print(clf.best_params_)
    print(clf.best_estimator_)
    print(clf.best_score_)
    print(cv_results)
    # print(metrics.f1_score(Y_true,Y_pred,average='macro'))
    # print(clf.score(X_test, Y_true))
    # print(SVMSGDClassifier(max_epoch=1000,batch_size=30).fit(tr_x, tr_x_lbl).score(tt_x, tt_x_lbl))

def data_generator(dataset='D1', path='./dataset'):
    if dataset == 'D1':
        _img = os.path.join(path, 'train-images.idx3-ubyte')
        _lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "D2":
        _img = os.path.join(path, 'test-images.idx3-ubyte')
        _lbl = os.path.join(path, 'test-labels.idx1-ubyte')
    else:
        print("?")

    with open(_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows * cols)

    def get_img(idx): return np.concatenate(
        ([lbl[idx]], list(img[idx])), axis=0)

    for i in range(len(lbl)):
        yield get_img(i)


def load_dataset(dataset):
    data = list(data_generator(dataset))
    columns = ["label"] + [f'#{x}' for x in range(784)]
    df = pd.DataFrame(data, columns=columns)
    print(df.describe())
    X = df.iloc[:, 1:]
    Y = df.iloc[:, 0]
    X=[]
    for i in range(60000):
        X.append(np.ravel(X[i][1]))
    Y=[]
    for i in range(60000):
        Y.append(X[i][0])
    return X, Y

def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 'test-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'test-labels.idx1-ubyte')
    else:
        raise Exception("dataset must be 'testing' or 'training'")

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

if __name__ == '__main__':
    main()
