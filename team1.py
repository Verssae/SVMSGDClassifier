import argparse
import os
import struct
import pandas as pd
import numpy as np

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
    tr_x, tr_x_lbl = load_dataset(tr)
    tt_x, tt_x_lbl = load_dataset(tt)

def data_generator(dataset='D1', path='dataset'):
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

    return X, Y


if __name__ == '__main__':
    main()
