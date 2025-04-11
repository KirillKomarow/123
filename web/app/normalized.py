import numpy as np
import pandas as pd
from sklearn import preprocessing

def get_normalized_data(data):
    scaler = preprocessing.MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data
    