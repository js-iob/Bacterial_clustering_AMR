#Author: K. T. Shreya Parthasarathi
#Script:  Use multi layer perceptron for prediction on multi-label classification

from numpy import asarray
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np



# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

# load dataset
df = pd.read_csv('classification_input_021524.txt', sep = '\t')
df = df.drop(['header'], axis =1)
#df["Cluster"] = df["Cluster"].astype(str)
print (df.dtypes)

X = np.asarray(df[df.columns[0:7]])
print (X.shape)

y = np.asarray(df[df.columns[7:]])
print (y.shape)

n_inputs, n_outputs = X.shape[1], y.shape[1]
# get model
model = get_model(n_inputs, n_outputs)
# fit the model on all data
model.fit(X, y, verbose=0, epochs=100)
# make a prediction for new data
row = [3940614,39.16831235,1198243,1198899,780917,762555,6]
newX = asarray([row])
yhat = model.predict(newX)
print('Predicted: %s' % yhat[0])
