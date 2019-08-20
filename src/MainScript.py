#from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import DataPreprocessing
import ModelMgr
import Config



def main():
    # TrainData: Includes the item outlet sales for each item
    # This data will be used to learn and test out ML algorithm to predict item outlet sales
    TrainData = pd.read_csv('../res/TrainData.txt', sep=",")

    # TestData is the data without any item outlet sales
    # This data is used to evaluate our ML algorithm
    TestData = pd.read_csv('../res/TestData.txt', sep=",")

    # Prepare train and test data to be used in our ML algorithm
    x_train, x_test, y_train, y_test = DataPreprocessing.PrepareData(TrainData)

    #Build model
    Model = ModelMgr.BuildModel(x_train)
    Model.summary()

    ModelCallBack = ModelMgr.PrintDot(x_train.shape[0])

    #Train model
    history = Model.fit(
        x_train, y_train,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS, validation_split=Config.VALIDATION_SPLIT, verbose=0,
        callbacks=[ModelCallBack])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())

    ModelMgr.plot_history(history)


if __name__ == "__main__":
    main()



