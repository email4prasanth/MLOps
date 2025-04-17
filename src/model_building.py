import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

# DataFrame
train_data = pd.read_csv(r"./data/processed/train_processed.csv")
# Splitting Data to features and target and data to numpy arrays (.values) work with positions
# features/Independent (X = df.iloc[:, 0:-1]) and target/Dependent var (y = df.iloc[:, -1]) 
# x_train = train_data.iloc[:,0:-1].values # Select all rows and all coulumns except last one
# y_train = train_data.iloc[:,-1].values # Select all rows and only last column

# keeps data as pandas objects DataFrame/Series explicitly specify the target by name 
x_train = train_data.drop(columns = ["Potability"], axis=1)
y_train = train_data["Potability"]

# Classifier
clf = RandomForestClassifier()
clf.fit(x_train,y_train)

pickle.dump(clf, open("model.pkl","wb")) #write binary