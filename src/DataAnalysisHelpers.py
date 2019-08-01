import pandas as pd
import numpy as np
import matplotlib as plt
import re

def PrepareData(TrainData, TestData):
    # Tests
    #head = data.head()

    #cols = data.columns

    #Weight = data.Item_Weight

    #NullWeigths = data[data.Item_Weight.isnull()]

    # End Tests

    # Drop the rows where at least one element is missing.
    TrainDataNotNull = TrainData.dropna()

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

    #Put all labels In lower or upper case
    TrainDataDropCategorical = pd.get_dummies(TrainDataLowerCase, columns=['Item_Fat_Content', 'Outlet_Type', 'Outlet_Location_Type', 'Outlet_Identifier', 'Item_Type', 'Outlet_Size'])

    #Drop identifier
    TrainDataForMLAlgo = TrainDataDropCategorical.drop(['Item_Identifier'], axis=1)

    return TrainDataForMLAlgo, TestData