#Author: K. T. Shreya Parthasarathi
#Script: Use multi-layer perceptron for multi-label classification

import pandas as pd
import numpy as np
from sklearn.datasets import make_multilabel_classification

df = pd.read_csv('classification_input_021524.txt', sep = '\t')
df = df.drop(['header'], axis =1)
#df["Cluster"] = df["Cluster"].astype(str)
print (df.dtypes)


X = np.asarray(df[df.columns[0:7]])
print (X.shape)

y = np.asarray(df[df.columns[7:]])
print (y.shape)

for i in range(10):
	print(X[i], y[i])

from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
 

# get the model
def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

from sklearn.metrics import hamming_loss
#print('Hamming Loss: ', round(hamming_loss(y_test, prediction),2))

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
    results = list()
    hamming = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
    # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # define model
        model = get_model(n_inputs, n_outputs)
        # fit model
        model.fit(X_train, y_train, verbose=0, epochs=100)
        # make a prediction on the test set
        yhat = model.predict(X_test)
        # round probabilities to class labels
        yhat = yhat.round()
        # calculate accuracy
        acc = accuracy_score(y_test, yhat)
        # store result
        ham = hamming_loss(y_test,yhat)
        print('>%.3f' % acc)
        print('Hamming Loss: ', ham)
        results.append(acc)
        hamming.append(ham)
    return results, hamming
        
 

# evaluate model
results = evaluate_model(X, y)
hamming = evaluate_model(X, y)
# summarize performance
print('Accuracy: %.3f (%.3f)' % (mean(results), std(results)))
print ('Hamming loss:', round(mean(hamming),2))

