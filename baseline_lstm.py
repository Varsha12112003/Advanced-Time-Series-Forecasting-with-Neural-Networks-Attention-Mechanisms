"""
baseline_lstm.py
Baseline many-to-many LSTM (predict horizon steps) — saves predictions and truth arrays.
"""
import argparse, numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from preprocessing import create_windows, scale_train_val_test

def build_model(window_size, features, horizon, hidden=128):
    model = Sequential([
        LSTM(hidden, input_shape=(window_size, features), return_sequences=False),
        Dense(horizon * features)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='synthetic.npy')
    p.add_argument('--window', type=int, default=60)
    p.add_argument('--horizon', type=int, default=5)
    p.add_argument('--epochs', type=int, default=30)
    args = p.parse_args()

    data = np.load(args.data)
    n = len(data)
    train, val, test = data[:int(0.7*n)], data[int(0.7*n):int(0.85*n)], data[int(0.85*n):]
    Xtr, ytr = create_windows(train, args.window, args.horizon)
    Xv, yv = create_windows(val, args.window, args.horizon)
    Xt, yt = create_windows(test, args.window, args.horizon)
    Xtr_s, Xv_s, Xt_s, scaler = scale_train_val_test(Xtr, Xv, Xt)

    ytr_flat = ytr.reshape((ytr.shape[0], -1))
    yv_flat = yv.reshape((yv.shape[0], -1))
    yt_flat = yt.reshape((yt.shape[0], -1))

    model = build_model(args.window, data.shape[1], args.horizon)
    model.fit(Xtr_s, ytr_flat, validation_data=(Xv_s, yv_flat), epochs=args.epochs, batch_size=64)
    model.save('baseline_lstm.h5')
    preds = model.predict(Xt_s).reshape((Xt_s.shape[0], args.horizon, data.shape[1]))
    np.save('baseline_preds.npy', preds)
    np.save('baseline_truth.npy', yt)
    print('Saved baseline model and predictions')

