"""
__author__ = "Param Popat"
__version__ = "1"
__git__ = "https://github.com/parampopat/"
"""


import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import *
from sklearn.metrics import accuracy_score


def get_model(xtrain, ytrain, xtest, ytest):
    """
    Trains and returns the model
    :return: Trained Model
    """
    model = Sequential()
    model.add(Dense(512, input_shape=(xtrain.shape[1],), activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=10, batch_size=128, verbose=2)
    return model


def train():
    """
    Main function to train the model
    :return:
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    print(X_train.shape[1])
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    digits = 10

    y_train = to_categorical(y_train, digits)
    y_test = to_categorical(y_test, digits)
    model = get_model(X_train, y_train, X_test, y_test)
    y_test_pred = model.predict(X_test)
    acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Acc:', acc[1])
    return model


trained_model = train()