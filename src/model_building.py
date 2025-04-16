import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv(r"./data/processed/train_processed.csv")
# Splitting features/Independent (X = df.iloc[:, 0:-1]) and target/Dependent var (y = df.iloc[:, -1]) 
x_train = train_data.iloc[:,0:-1].values # Select all rows and all coulumns except last one
y_train = train_data.iloc[:,-1].values # Select all rows and only last column

# Classifier
clf = RandomForestClassifier()
clf.fit(x_train,y_train)

pickle.dump(clf, open("model.pkl","wb")) #write binary