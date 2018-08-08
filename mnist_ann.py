"""
__author__ = "Param Popat"
__version__ = "1"
__git__ = "https://github.com/parampopat/"
"""


from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import *


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
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    print(x_train.shape[1])
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    digits = 10
    y_train = to_categorical(y_train, digits)
    y_test = to_categorical(y_test, digits)
    model = get_model(x_train, y_train, x_test, y_test)
    y_test_pred = model.predict(x_test)
    acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test Acc:', acc[1])
    return model


trained_model = train()
