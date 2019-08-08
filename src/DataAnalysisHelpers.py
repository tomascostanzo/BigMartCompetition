import pandas as pd
import numpy as np
import matplotlib as plt
import re
from sklearn.model_selection import train_test_split

def PrepareData(Data):

    #TrainHead = Data.head()

    #print(TrainHead.columns)

    # Drop the rows where at least one element is missing.
    TrainDataNotNull = Data.dropna()

    #Remove special characters
    TrainDataWithoutSpecialCharacters = TrainDataNotNull.applymap(lambda s:s.replace('[^a-zA-Z0-9-_.]', '') if type(s) == str else s)

    #Put all labels In lower or upper case
    TrainDataLowerCase = TrainDataWithoutSpecialCharacters.applymap(lambda s:s.lower() if type(s) == str else s)

    abbr_dict={
        r"\breg\b":"regular",
        r"\blf\b":"low fat",
        }

    #Replace abbreviations
    TrainDataLowerCase.replace(abbr_dict, regex=True, inplace=True)

    #Add Age of supermarket
    TrainDataLowerCase['Supermarket_Age'] = TrainDataLowerCase.apply(lambda row: 2019 - row['Outlet_Establishment_Year'], axis=1)

    #Replace labels into one hot encodine
    TrainDataOneHotEcoding = pd.get_dummies(TrainDataLowerCase, columns=['Item_Fat_Content', 'Outlet_Type', 'Outlet_Location_Type', 'Outlet_Identifier', 'Item_Type', 'Outlet_Size'])

    #Drop identifier -> see if it is good
    TrainDataForMLAlgo = TrainDataOneHotEcoding.drop(['Item_Identifier'], axis=1)

    y = TrainDataForMLAlgo.pop('Item_Outlet_Sales')
    x = TrainDataForMLAlgo

    assert(len(x) == len(y))

    #Get stats
    x_stats = x.describe()
    x_stats = x_stats.transpose()

    #Normalize
    #Normalizing even one hot encoding, check if better not doing so
    normed_x = norm(x, x_stats)

    X_train, X_test, y_train, y_test = train_test_split(normed_x, y, test_size=0.33)

    return X_train, X_test, y_train, y_test


def norm(DataSet, DataSetStats):
    return (DataSet - DataSetStats['mean']) / DataSetStats['std']

#def norm2(DataSet, DataSetStats, ColumnsToNormalize)
 #   modDfObj = DataSet.apply(lambda x: (x - DataSetStats['mean']) / DataSetStats['std'] if x.name in ColumnsToNormalize else x)