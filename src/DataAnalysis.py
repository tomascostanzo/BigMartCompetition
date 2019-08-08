#from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from DataAnalysisHelpers import PrepareData

#TrainData: Includes the item outlet sales for each item
#This data will be used to learn and test out ML algorithm to predict item outlet sales
TrainData = pd.read_csv('res/TrainData.txt', sep=",")

#TestData is the data without any item outlet sales
#This data is used to evaluate our ML algorithm
TestData = pd.read_csv('res/TestData.txt', sep=",")

#Prepare train and test data to be used in our ML algorithm
x_train, x_test, y_train, y_test = PrepareData(TrainData)
sns.pairplot(x_train[["Item_Weight", "Item_MRP", "Item_Visibility"]], diag_kind="kde")
plt.show()

print(x_train)