"""preprocessing.py
Sequence windowing and scaling utilities.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_windows(data, window_size=60, horizon=1):
    # data: (timesteps, features)
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+horizon])
    return np.array(X), np.array(y)

def scale_train_val_test(train, val, test):
    # fit scaler on train only
    scaler = MinMaxScaler()
    scaler.fit(train.reshape(-1, train.shape[-1]))
    def transform(arr):
        s = arr.reshape(-1, arr.shape[-1])
        s = scaler.transform(s)
        return s.reshape(arr.shape)
    return transform(train), transform(val), transform(test), scaler
