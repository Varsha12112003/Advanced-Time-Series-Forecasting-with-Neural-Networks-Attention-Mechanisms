"""baseline_lstm.py
Simple LSTM baseline using TensorFlow/Keras.
"""
import argparse
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint
from preprocessing import create_windows, scale_train_val_test

def build_model(window_size, features, horizon, hidden=64):
    model = Sequential()
    model.add(LSTM(hidden, input_shape=(window_size, features), return_sequences=False))
    if horizon == 1:
        model.add(Dense(features))
    else:
        model.add(Dense(horizon*features))
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../assets/synthetic.npy')
    parser.add_argument('--window', type=int, default=60)
    parser.add_argument('--horizon', type=int, default=1)
    args = parser.parse_args()
    data = np.load(args.data)
    n = len(data)
    train_end = int(0.7*n)
    val_end = int(0.85*n)
    train, val, test = data[:train_end], data[train_end:val_end], data[val_end:]
    Xtr, ytr = create_windows(train, args.window, args.horizon)
    Xv, yv = create_windows(val, args.window, args.horizon)
    Xt, yt = create_windows(test, args.window, args.horizon)
    Xtr_s, Xv_s, Xt_s, scaler = scale_train_val_test(Xtr, Xv, Xt)
    # reshape y for horizon>1 simple flattening (user can adapt to many-to-many)
    if args.horizon > 1:
        ytr = ytr.reshape((ytr.shape[0], -1))
        yv = yv.reshape((yv.shape[0], -1))
    else:
        ytr = ytr[:,0,:]
        yv = yv[:,0,:]
    model = build_model(args.window, data.shape[1], args.horizon)
    model.fit(Xtr_s, ytr, validation_data=(Xv_s, yv), epochs=5, batch_size=64)
    model.save('baseline_lstm.h5')
    print('Saved baseline_lstm.h5')
