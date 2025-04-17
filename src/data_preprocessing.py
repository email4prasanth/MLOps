import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# input DataFrame
train_data =  pd.read_csv("./data/raw/train.csv")
test_data =  pd.read_csv("./data/raw/test.csv")

# Median fuction to fill the missing values 
def missing_vlaues_meadian(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
    return df
train_processed_data = missing_vlaues_meadian(train_data)
test_processed_data = missing_vlaues_meadian(test_data)

# create dir to store the processed
data_path = os.path.join("data", "processed")
os.makedirs(data_path)

# Convert data to CSV and create csv file
train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"),index=False)
test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"),index=False)
