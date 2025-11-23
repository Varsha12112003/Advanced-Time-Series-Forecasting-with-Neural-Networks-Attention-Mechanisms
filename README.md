# Advanced Time Series Forecasting — Deliverables

Included files:
- README.md (this file)
- REPORT.md — concise project report and instructions
- code/
  - data_gen.py — generate synthetic multivariate time series
  - preprocessing.py — sequence windowing and scaling
  - baseline_lstm.py — simple LSTM baseline
  - seq2seq_attention.py — Seq2Seq model with Bahdanau attention (TensorFlow/Keras)
  - evaluate.py — evaluation metrics (RMSE, MAE, MAPE) and example run
  - requirements.txt — Python package requirements
- assets/
  - screenshot.png — the image you uploaded

How to use:
1. Create a Python virtual environment and install requirements:
   ```
   python -m venv venv
   source venv/bin/activate    # or venv\Scripts\activate on Windows
   pip install -r code/requirements.txt
   ```
2. Generate data:
   ```
   python code/data_gen.py --out ../assets/synthetic.npy
   ```
3. Train baseline:
   ```
   python code/baseline_lstm.py --data ../assets/synthetic.npy
   ```
4. Train seq2seq attention:
   ```
   python code/seq2seq_attention.py --data ../assets/synthetic.npy
   ```
5. Evaluate:
   ```
   python code/evaluate.py --pred preds.npy --truth truth.npy
   ```

Keep outputs and logs in a separate folder. Good luck!
