import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# DataFrame
data = pd.read_csv(r"D:\PersonalRepo\MLOPS\Lecture_Notes\Data_Collection\water_potability.csv")

train_data, test_data = train_test_split(data, test_size= 0.2, random_state = 40)

data_path = os.path.join("data", "raw")
os.makedirs(data_path)
train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_data.to_csv(os.path.join(data_path, "test.csv"), index= False)