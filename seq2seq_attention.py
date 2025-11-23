"""
seq2seq_attention.py
Seq2Seq with Bahdanau Attention supporting teacher forcing and true autoregressive decoding.
Also includes attention weight extraction and optional saving of attention maps.
"""
import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Layer
from tensorflow.keras.models import Model
from preprocessing import create_windows, scale_train_val_test

# -----------------------------
# Attention layer
# -----------------------------
class BahdanauAttention(Layer):
    def __init__(self, units, name='bahdanau_attention'):
        super(BahdanauAttention, self).__init__(name=name)
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

# -----------------------------
# Training graph (teacher forcing)
# -----------------------------
def build_seq2seq(window, features, horizon, latent=128):
    # Encoder
    enc_inputs = Input(shape=(window, features), name='enc_input')
    enc_lstm = LSTM(latent, return_sequences=True, return_state=True, name='enc_lstm')
    enc_out, enc_h, enc_c = enc_lstm(enc_inputs)

    # Decoder (teacher-forcing)
    dec_inputs = Input(shape=(horizon, features), name='dec_input')
    dec_lstm = LSTM(latent, return_state=True, return_sequences=True, name='dec_lstm')
    dec_outputs_seq, _, _ = dec_lstm(dec_inputs, initial_state=[enc_h, enc_c])  # (batch, horizon, latent)

    attention = BahdanauAttention(latent, name='bahdanau_attention')
    dense = Dense(features, name='out_dense')

    # Apply attention at each decoder step
    outputs = []
    attn_weights_steps = []
    for t in range(horizon):
        dec_h_t = dec_outputs_seq[:, t, :]               # (batch, latent)
        context_t, weights_t = attention(dec_h_t, enc_out)
        out_t = dense(context_t)                         # (batch, features)
        outputs.append(out_t)
        attn_weights_steps.append(weights_t)             # (batch, window)

    outputs = tf.stack(outputs, axis=1)                  # (batch, horizon, features)
    # Stack attention weights for potential inspection in training graph
    attn_weights = tf.stack(attn_weights_steps, axis=1)  # (batch, horizon, window)

    model = Model([enc_inputs, dec_inputs], outputs, name='seq2seq_attn')
    model.compile(optimizer='adam', loss='mse')
    return model

# -----------------------------
# Helper: build teacher-forcing inputs
# -----------------------------
def make_decoder_inputs(y):
    # shift-right of target with zero first frame
    dec = np.zeros_like(y)
    dec[:, 1:, :] = y[:, :-1, :]
    return dec

# -----------------------------
# Autoregressive inference (true iterative decoding)
# -----------------------------
def build_encoder_decoder_submodels(full_model):
    # Encoder submodel: input -> encoder outputs, states
    enc_input = full_model.get_layer('enc_input').input
    enc_out, enc_h, enc_c = full_model.get_layer('enc_lstm').output
    encoder = Model(enc_input, [enc_out, enc_h, enc_c], name='encoder_infer')

    # Attention and dense layers
    attention_layer = full_model.get_layer('bahdanau_attention')
    out_dense = full_model.get_layer('out_dense')

    # Decoder one-step submodel:
    # Inputs: decoder previous input (1, features), previous states, encoder outputs
    dec_input_step = Input(shape=(1, enc_h.shape[-1]), name='dec_input_step_features')  # we will feed predicted frame but model expects features size
    # However our training graph uses features as decoder input.
    # For one-step LSTM we need a decoder LSTM layer with return_state=True and return_sequences=False.
    # Recreate a decoder cell with same weights by referencing the trained layer.
    dec_lstm_layer = full_model.get_layer('dec_lstm')
    # Keras Trick: call the layer on a step input to reuse weights
    # Note: we will pass states each step.

    # Placeholders for previous states
    dec_h_in = Input(shape=(enc_h.shape[-1],), name='dec_h_in')
    dec_c_in = Input(shape=(enc_c.shape[-1],), name='dec_c_in')
    # Run one step (sequence length 1)
    dec_out_step, dec_h_out, dec_c_out = dec_lstm_layer(dec_input_step, initial_state=[dec_h_in, dec_c_in])

    # Attention over encoder outputs using current hidden state
    enc_outputs_in = Input(shape=(enc_out.shape[1], enc_out.shape[2]), name='enc_outputs_in')
    context_t, attn_weights_t = attention_layer(dec_out_step, enc_outputs_in)
    pred_t = out_dense(context_t)

    decoder_step = Model(
        [dec_input_step, dec_h_in, dec_c_in, enc_outputs_in],
        [pred_t, dec_h_out, dec_c_out, attn_weights_t],
        name='decoder_step'
    )
    return encoder, decoder_step

def autoregressive_predict(full_model, Xt_s, horizon, features):
    encoder, decoder_step = build_encoder_decoder_submodels(full_model)
    preds = np.zeros((Xt_s.shape[0], horizon, features), dtype=np.float32)
    attn_maps = np.zeros((Xt_s.shape[0], horizon, Xt_s.shape[1]), dtype=np.float32)  # window length = Xt_s.shape[1]

    # Encode all windows
    enc_out, enc_h, enc_c = encoder.predict(Xt_s, verbose=0)

    # Iterate per sample
    for i in range(Xt_s.shape[0]):
        # Start token: zeros
        dec_input_t = np.zeros((1, 1, features), dtype=np.float32)
        h_t = enc_h[i:i+1]
        c_t = enc_c[i:i+1]
        enc_out_i = enc_out[i:i+1]

        for t in range(horizon):
            y_t, h_t, c_t, w_t = decoder_step.predict([dec_input_t, h_t, c_t, enc_out_i], verbose=0)
            preds[i, t, :] = y_t[0]
            attn_maps[i, t, :] = w_t[0]
            # Next input is the predicted frame
            dec_input_t = y_t.reshape(1, 1, features)

    return preds, attn_maps

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='synthetic.npy')
    parser.add_argument('--window', type=int, default=60)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--save_attn', action='store_true', help='Save attention maps during inference')
    parser.add_argument('--outdir', default='outputs', help='Directory to save models and npy files')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load and split
    data = np.load(args.data)
    n = len(data)
    train, val, test = data[:int(0.7*n)], data[int(0.7*n):int(0.85*n)], data[int(0.85*n):]
    Xtr, ytr = create_windows(train, args.window, args.horizon)
    Xv, yv = create_windows(val, args.window, args.horizon)
    Xt, yt = create_windows(test, args.window, args.horizon)

    # Scale
    Xtr_s, Xv_s, Xt_s, scaler = scale_train_val_test(Xtr, Xv, Xt)

    # Teacher forcing inputs
    dec_tr = make_decoder_inputs(ytr)
    dec_v = make_decoder_inputs(yv)

    # Build and train
    model = build_seq2seq(args.window, data.shape[1], args.horizon, latent=128)
    history = model.fit(
        [Xtr_s, dec_tr], ytr,
        validation_data=([Xv_s, dec_v], yv),
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=1
    )

    # Save model
    model_path = os.path.join(args.outdir, 'seq2seq_attention.h5')
    model.save(model_path)

    # True autoregressive inference
    preds, attn_maps = autoregressive_predict(model, Xt_s, args.horizon, data.shape[1])

    # Save predictions and truth
    preds_path = os.path.join(args.outdir, 'seq2seq_preds.npy')
    truth_path = os.path.join(args.outdir, 'seq2seq_truth.npy')
    np.save(preds_path, preds)
    np.save(truth_path, yt)

    # Optionally save attention maps
    if args.save_attn:
        attn_path = os.path.join(args.outdir, 'seq2seq_attn_maps.npy')
        np.save(attn_path, attn_maps)

    # Print summary
    print(f'Saved model: {model_path}')
    print(f'Saved predictions: {preds_path}')
    print(f'Saved ground truth: {truth_path}')
    if args.save_attn:
        print(f'Saved attention maps: {attn_path}')

    # Tip for evaluation: run evaluate.py after this script.
    # Example:
    #   python evaluate.py --pred {preds_path} --truth {truth_path} --metrics rmse mae mape



