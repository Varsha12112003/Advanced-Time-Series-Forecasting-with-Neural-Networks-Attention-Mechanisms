# Project Report — Advanced Time Series Forecasting (concise)

## Objective
Build and compare a baseline LSTM and a Seq2Seq model with attention for multivariate time series forecasting. Provide code, preprocessing, training scripts, and evaluation (RMSE, MAE, MAPE).

## Data
- Synthetic multivariate dataset (5 features, 5000 timesteps) included generator script.
- Saved as NumPy array with shape (timesteps, features).

## Pipeline
1. Data generation → normalization (MinMax) → sliding window sequence creation.
2. Time-aware train/val/test split by contiguous segments (no shuffling).
3. Baseline LSTM: many-to-one or many-to-many depending on horizon.
4. Seq2Seq + Bahdanau attention: encoder LSTM, decoder LSTM with attention layer over encoder outputs.
5. Hyperparameter knobs: window_size, forecast_horizon, batch_size, hidden_units, learning_rate.
6. Evaluation: RMSE, MAE, MAPE across horizons.

## Findings (example expected)
- Baseline LSTM is strong for short horizons; attention model improves multi-step forecasting and interpretable temporal/feature attention weights.
- Use attention weights to inspect which timesteps and features contributed to predictions.

## Limitations
- Synthetic data may not capture real-world noise. For production use, test on real datasets (electricity, energy, finance).
- Further improvements: Transformer-based models, probabilistic forecasting, ensembling.

## Files included
See README.md for usage.
