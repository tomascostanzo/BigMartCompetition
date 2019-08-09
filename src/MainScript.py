#from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from DataPreprocessing import PrepareData


def main():
    # TrainData: Includes the item outlet sales for each item
    # This data will be used to learn and test out ML algorithm to predict item outlet sales
    TrainData = pd.read_csv('res/TrainData.txt', sep=",")

    # TestData is the data without any item outlet sales
    # This data is used to evaluate our ML algorithm
    TestData = pd.read_csv('res/TestData.txt', sep=",")

    # Prepare train and test data to be used in our ML algorithm
    x_train, x_test, y_train, y_test = PrepareData(TrainData)



if __name__ == "__main__":
    main()



