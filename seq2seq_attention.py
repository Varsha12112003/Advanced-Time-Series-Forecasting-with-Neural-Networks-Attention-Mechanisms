"""seq2seq_attention.py
Seq2Seq with Bahdanau attention (Keras). This is a compact, instructional implementation.
"""
import argparse
import numpy as np
from tensorflow.keras.layers import Input, LSTM, Dense, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from preprocessing import create_windows, scale_train_val_test

class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        # query: decoder hidden state (batch, hidden)
        # values: encoder outputs (batch, timesteps, hidden)
        query_with_time_axis = K.expand_dims(query, 1)
        score = K.tanh(self.W1(values) + self.W2(query_with_time_axis))
        attention_weights = K.softmax(self.V(score), axis=1)
        context_vector = attention_weights * values
        context_vector = K.sum(context_vector, axis=1)
        return context_vector, attention_weights

def build_model(window_size, features, latent=64, horizon=1):
    # Encoder
    encoder_inputs = Input(shape=(window_size, features))
    encoder_lstm = LSTM(latent, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    # Decoder (we use one-step decoding repeated for horizon simple case)
    decoder_inputs = Input(shape=(1, features))
    decoder_lstm = LSTM(latent, return_sequences=False, return_state=True)
    dense = Dense(features)
    attention = BahdanauAttention(latent)
    # training-time simplification: run decoder for single step prediction of next horizon steps flattened
    # For clarity, we'll predict horizon=1
    context, attn = attention(state_h, encoder_outputs)
    # combine context (batch, latent) to produce output
    outputs = dense(context)
    model = Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../assets/synthetic.npy')
    parser.add_argument('--window', type=int, default=60)
    args = parser.parse_args()
    data = np.load(args.data)
    n = len(data)
    train_end = int(0.7*n)
    val_end = int(0.85*n)
    train, val, test = data[:train_end], data[train_end:val_end], data[val_end:]
    Xtr, ytr = create_windows(train, args.window, 1)
    Xv, yv = create_windows(val, args.window, 1)
    Xt, yt = create_windows(test, args.window, 1)
    Xtr_s, Xv_s, Xt_s, scaler = scale_train_val_test(Xtr, Xv, Xt)
    # decoder dummy inputs (zeros) for training simplification
    dec_tr = np.zeros((Xtr_s.shape[0], 1, Xtr_s.shape[2]))
    dec_v = np.zeros((Xv_s.shape[0], 1, Xv_s.shape[2]))
    model = build_model(args.window, data.shape[1])
    model.fit([Xtr_s, dec_tr], ytr[:,0,:], validation_data=([Xv_s, dec_v], yv[:,0,:]), epochs=5, batch_size=64)
    model.save('seq2seq_attention.h5')
    print('Saved seq2seq_attention.h5')
