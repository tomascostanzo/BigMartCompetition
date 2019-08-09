import pandas as pd
import numpy as np
import matplotlib as plt
import re
from sklearn.model_selection import train_test_split

def PrepareData(RawData):

    Data = RawData.copy()

    # Drop the rows where at least one element is missing.
    Data = Data.dropna()

    #Remove special characters
    Data  = Data.applymap(lambda s:s.replace('[^a-zA-Z0-9-_.]', '') if type(s) == str else s)

    #Put all labels In lower or upper case
    Data = Data.applymap(lambda s:s.lower() if type(s) == str else s)

    #Drop identifier -> see if it is good
    Data = Data.drop(['Item_Identifier'], axis=1)

    #Here find manually all similar abbreviations
    abbr_dict={
        r"\breg\b":"regular",
        r"\blf\b":"low fat",
        }

    #Replace abbreviations
    Data.replace(abbr_dict, regex=True, inplace=True)

    #Replace labels into one hot encodine
    Data = pd.get_dummies(Data, columns=['Item_Fat_Content', 'Outlet_Type', 'Outlet_Location_Type', 'Outlet_Identifier', 'Item_Type', 'Outlet_Size'])

    #Add other features
    Data = AddOtherFeaturesManually(Data)

    #Separe the output (predictions) and input data
    y = Data.pop('Item_Outlet_Sales')
    x = Data

    assert(len(x) == len(y))

    #Get stats
    x_stats = x.describe()
    x_stats = x_stats.transpose()

    #Normalize
    #Normalizing even one hot encoding, check if better not doing so
    normed_x = normalizeSelectedColumns(x, x_stats, ['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year','Supermarket_Age'])

    x_train, x_test, y_train, y_test = train_test_split(normed_x, y, test_size=0.33)

    return x_train, x_test, y_train, y_test


def normalizeSelectedColumns(DataSet, DataSetStats, ColumnsToNormalize):
    return DataSet.apply(lambda x: (x - DataSetStats['mean'][x.name]) / DataSetStats['std'][x.name] if x.name in ColumnsToNormalize else x, axis=0)

def AddOtherFeaturesManually(Data):
    Data['Supermarket_Age'] = Data.apply(lambda row: 2019 - row['Outlet_Establishment_Year'], axis=1)
    return Data

"""
Help functions:
TrainHead = Data.head()
print(TrainHead.columns)
"""