import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

test_data = pd.read_csv(r"./data/processed/test_processed.csv")
x_test = test_data.iloc[:,0:-1].values
y_test = test_data.iloc[:,-1].values

model = pickle.load(open("model.pkl","rb"))
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1s = f1_score(y_test, y_pred)

Metric_dict ={
    'accuacy': acc,
    'precision': pre,
    'recall': rec,
    'f1sc': f1s
}

with open('metrics.json','w') as file:
    json.dump(Metric_dict,file,indent=4)
