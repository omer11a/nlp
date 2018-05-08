#!/usr/bin/python3

import sys
import sklearn.datasets
import sklearn.linear_model
import pickle

def main(feature_vector_filename, model_filename):
    x_train, y_train = sklearn.datasets.load_svmlight_file(feature_vector_filename)
    model = sklearn.linear_model.LogisticRegression()
    model.fit(x_train, y_train)
    pickle.dump(model, open(model_filename, 'wb'))

if __name__ == '__main__':
    main(*sys.argv[1:])
