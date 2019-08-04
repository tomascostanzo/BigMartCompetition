import pandas as pd
import numpy as np
import matplotlib as plt
import re
from sklearn.model_selection import train_test_split

def PrepareData(Data):


    TestHead = TestData.head()
    TrainHead = Data.head()

    print(TestHead.columns)
    print(TrainHead.columns)

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

    #REplace labels into one hot encodine
    TrainDataOneHotEcoding = pd.get_dummies(TrainDataLowerCase, columns=['Item_Fat_Content', 'Outlet_Type', 'Outlet_Location_Type', 'Outlet_Identifier', 'Item_Type', 'Outlet_Size'])

    #Drop identifier -> see if it is good
    TrainDataForMLAlgo = TrainDataOneHotEcoding.drop(['Item_Identifier'], axis=1)

    y = TrainDataForMLAlgo.pop('Item_Outlet_Sales')
    x = TrainDataForMLAlgo

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)


    return TrainDataForMLAlgo