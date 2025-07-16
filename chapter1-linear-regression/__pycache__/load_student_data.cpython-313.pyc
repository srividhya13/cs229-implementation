import pandas as pd 
import numpy as np
def load_student_data():
    df=pd.read_csv("StudentsPerformance.csv")
    X = df[["reading score", "writing score"]].values
    y = df["math score"].values.reshape(-1, 1)
    X_b = np.c_[np.ones((X.shape[0], 1)), X] 
    return X_b, y 