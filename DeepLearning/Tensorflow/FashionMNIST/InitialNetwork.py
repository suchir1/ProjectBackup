import tensorflow as tf
import numpy as np
import mnist_reader
import os
import sys

filepath = sys.argv[0][:sys.argv[0].rfind("/")]+"/data/fashion"
x_train, y_train = mnist_reader.load_mnist(filepath, kind='train')
x_test, y_test = mnist_reader.load_mnist(filepath, kind='t10k')

