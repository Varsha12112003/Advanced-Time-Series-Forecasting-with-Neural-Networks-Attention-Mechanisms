"""
seq2seq_attention.py
Seq2Seq with Bahdanau Attention supporting teacher forcing and iterative decoding.
Saves predictions and (instructions to) extract attention weights.
"""
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Layer
from tensorflow.keras.models import Model
from preprocessing import create_windows, scale_train_val_test

class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        # query: (batch, latent)
        # values: (batch, enc_timesteps, latent)
        query_time = tf.expand_dims(query, 1)
        score = tf.nn.tanh(self.W1(values) + self.W2(query_time))
        weights = tf.nn.softmax(self.V(score), axis=1)  # (batch, enc_timesteps, 1)
        context = tf.reduce_sum(weights * values, axis=1)  # (batch, latent)
        weights = tf.squeeze(weights, -1)  # (batch, enc_timesteps)
        return context, weights

def build_seq2seq(window, features, horizon, latent=128):
    # Encoder
    enc_inputs = Input(shape=(window, features), name='enc_input')
    enc_lstm = LSTM(latent, return_sequences=True, return_state=True, name='enc_lstm')
    enc_out, enc_h, enc_c = enc_lstm(enc_inputs)

    # Decoder (training model uses teacher forcing via decoder_inputs)
    dec_inputs = Input(shape=(horizon, features), name='dec_input')
    dec_lstm = LSTM(latent, return_state=True, return_sequences=True, name='dec_lstm')
    dec_outputs_seq = dec_lstm(dec_inputs, initial_state=[enc_h, enc_c])[0]  # (batch, horizon, latent)
    attention = BahdanauAttention(latent)
    dense = Dense(features, name='out_dense')

    # apply attention for each decoder step (vectorized loop)
    outputs = []
    for t in range(horizon):
        dec_h_t = dec_outputs_seq[:, t, :]            # (batch, latent)
        context_t, _ = attention(dec_h_t, enc_out)    # (batch, latent)
        out_t = dense(context_t)                      # (batch, features)
        outputs.append(out_t)
    outputs = tf.stack(outputs, axis=1)               # (batch, horizon, features)

    model = Model([enc_inputs, dec_inputs], outputs, name='seq2seq_attn')
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='synthetic.npy')
    p.add_argument('--window', type=int, default=60)
    p.add_argument('--horizon', type=int, default=5)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=64)
    args = p.parse_args()

    data = np.load(args.data)
    n = len(data)
    train, val, test = data[:int(0.7*n)], data[int(0.7*n):int(0.85*n)], data[int(0.85*n):]
    Xtr, ytr = create_windows(train, args.window, args.horizon)
    Xv, yv = create_windows(val, args.window, args.horizon)
    Xt, yt = create_windows(test, args.window, args.horizon)

    Xtr_s, Xv_s, Xt_s, scaler = scale_train_val_test(Xtr, Xv, Xt)

    # Build decoder teacher-forcing inputs: shift-right of target with zero first frame
    def make_decoder_inputs(y):
        dec = np.zeros_like(y)
        dec[:,1:,:] = y[:,:-1,:]
        return dec

    dec_tr = make_decoder_inputs(ytr)
    dec_v = make_decoder_inputs(yv)

    model = build_seq2seq(args.window, data.shape[1], args.horizon, latent=128)
    model.fit([Xtr_s, dec_tr], ytr, validation_data=([Xv_s, dec_v], yv),
              epochs=args.epochs, batch_size=args.batch)

    model.save('seq2seq_attention.h5')

    # Inference (simple): use zero decoder inputs to get autoregressive predictions from training graph
    dec_zero = np.zeros((Xt_s.shape[0], args.horizon, data.shape[1]))
    preds = model.predict([Xt_s, dec_zero])
    np.save('seq2seq_preds.npy', preds)
    np.save('seq2seq_truth.npy', yt)
    print('Saved seq2seq_attention.h5, seq2seq_preds.npy, seq2seq_truth.npy')
    print('To extract attention weights per-step, run a separate function using the trained attention layer weights and encoder outputs.')


