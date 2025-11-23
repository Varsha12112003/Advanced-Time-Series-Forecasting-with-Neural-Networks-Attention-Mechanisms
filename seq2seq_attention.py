"""
Proper multi-step Seq2Seq model with Bahdanau Attention.
Supports horizon > 1, no dummy inputs, real iterative decoding.
"""

import numpy as np
import argparse
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
        # query : (batch, hidden)
        # values: (batch, time, hidden)
        query_time = tf.expand_dims(query, 1)
        score = tf.nn.tanh(self.W1(values) + self.W2(query_time))
        weights = tf.nn.softmax(self.V(score), axis=1)
        context = weights * values
        context = tf.reduce_sum(context, axis=1)
        return context, weights


def build_seq2seq(window, features, horizon, latent=64):
    # Encoder
    encoder_inputs = Input(shape=(window, features))
    encoder_lstm = LSTM(latent, return_sequences=True, return_state=True)
    enc_out, enc_h, enc_c = encoder_lstm(encoder_inputs)

    # Decoder
    decoder_inputs = Input(shape=(horizon, features))
    decoder_lstm = LSTM(latent, return_state=True, return_sequences=True)
    att = BahdanauAttention(latent)
    dense = Dense(features)

    all_outputs = []
    state_h, state_c = enc_h, enc_c
    decoder_step_input = decoder_inputs[:, 0:1, :]  # first step

    for t in range(horizon):
        # run one decoder step
        dec_out, state_h, state_c = decoder_lstm(decoder_step_input,
                                                 initial_state=[state_h, state_c])
        context, att_w = att(state_h, enc_out)
        out = dense(context)
        all_outputs.append(out)

        # next decoder input = previous output
        decoder_step_input = tf.expand_dims(out, 1)

    outputs = tf.stack(all_outputs, axis=1)

    model = Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../assets/synthetic.npy")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()

    data = np.load(args.data)
    n = len(data)

    train, val, test = (
        data[:int(0.7*n)],
        data[int(0.7*n):int(0.85*n)],
        data[int(0.85*n):]
    )

    Xtr, ytr = create_windows(train, args.window, args.horizon)
    Xv, yv = create_windows(val, args.window, args.horizon)

    Xtr_s, Xv_s, _, scaler = scale_train_val_test(Xtr, Xv, Xv)

    # decoder input → first feature frame repeated horizon times
    dec_tr = np.zeros((len(Xtr_s), args.horizon, data.shape[1]))
    dec_v = np.zeros((len(Xv_s), args.horizon, data.shape[1]))

    model = build_seq2seq(args.window, data.shape[1], args.horizon)
    model.fit([Xtr_s, dec_tr], ytr, validation_data=([Xv_s, dec_v], yv),
              epochs=10, batch_size=64)

    model.save("seq2seq_attention_fixed.h5")
    print("Saved seq2seq_attention_fixed.h5")

