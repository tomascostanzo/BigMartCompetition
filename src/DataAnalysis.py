import pandas as pd
import numpy as np
import matplotlib as plt
import re
from DataAnalysisHelpers import PrepareData

TrainData = pd.read_csv('res/TrainData.txt', sep=",")
TestData = pd.read_csv('res/TestData.txt', sep=",")


TrainDataForMLAlgo = PrepareData(TrainData)
