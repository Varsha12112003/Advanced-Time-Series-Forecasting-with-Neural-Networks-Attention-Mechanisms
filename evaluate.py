"""evaluate.py
Simple evaluation metrics: RMSE, MAE, MAPE
"""
import numpy as np
import argparse
def rmse(a,b): return np.sqrt(np.mean((a-b)**2))
def mae(a,b): return np.mean(np.abs(a-b))
def mape(a,b): return np.mean(np.abs((a-b)/(b+1e-8)))*100

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True)
    parser.add_argument('--truth', required=True)
    args = parser.parse_args()
    p = np.load(args.pred)
    t = np.load(args.truth)
    print('RMSE:', rmse(p,t))
    print('MAE:', mae(p,t))
    print('MAPE:', mape(p,t))
