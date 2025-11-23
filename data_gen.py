"""data_gen.py
Generate a synthetic multivariate time series and save as numpy file.
"""
import numpy as np
import argparse
def generate(ts=5000, features=5, seed=42):
    rng = np.random.RandomState(seed)
    t = np.arange(ts)
    data = []
    for f in range(features):
        freq = 0.01*(f+1)
        phase = rng.rand()*2*np.pi
        trend = 0.0001*(f+1)*t
        noise = 0.1 * rng.randn(ts)
        series = np.sin(2*np.pi*freq*t + phase) + trend + noise
        data.append(series)
    return np.vstack(data).T  # shape (ts, features)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='../assets/synthetic.npy')
    parser.add_argument('--timesteps', type=int, default=5000)
    parser.add_argument('--features', type=int, default=5)
    args = parser.parse_args()
    X = generate(ts=args.timesteps, features=args.features)
    np.save(args.out, X)
    print('Saved synthetic data to', args.out)
