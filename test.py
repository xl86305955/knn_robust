import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
ones = [x_train[i] for i in range(len(y_train)) if y_train[i] == 1]
sevens = [x_train[i] for i in range(len(y_train)) if y_train[i] == 7]
onesTest = [x_test[i] for i in range(len(y_test)) if y_test[i] == 1]
sevensTest = [x_test[i] for i in range(len(y_test)) if y_test[i] == 7]

