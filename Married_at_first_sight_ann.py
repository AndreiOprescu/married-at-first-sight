# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

dataset = pd.read_csv("mafs.csv")
X = dataset.drop(["Status", "Couple", "Season", "Name"], axis=1).values
y = dataset.iloc[:, 8:9].values

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

#encode_columns = [0, 2, 3, 4, 5]
#for c in encode_columns:
#    lb_X = LabelEncoder()
#    X[:, c] = lb_X.fit_transform(X[:, c])
#
#lb_y = LabelEncoder()
#y = lb_y.fit_transform(y)
#
#ohe = OneHotEncoder(categorical_features = encode_columns)
#X = ohe.fit_transform(X).toarray()

ct_X = ColumnTransformer(transformers=[('en_X', OneHotEncoder(), [0, 2, 3, 4, 5])], remainder='passthrough')
X = ct_X.fit_transform(X).toarray()

lb_y = LabelEncoder()
y = lb_y.fit_transform(y)

sc_X = MinMaxScaler(feature_range = (0, 1))
X = sc_X.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from keras.models import Sequential
from keras.layers import Dense, Dropout

classifier = Sequential()

classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu', input_dim = 90))

classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, epochs = 50, batch_size = 16)

y_pred = classifier.predict(X_test)
y_pred = y_pred > 0.5