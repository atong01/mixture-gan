"""
Builds runs table for snakemake
"""
import numpy as np
import os
import pandas as pd


def build_runs():
    seeds = 20
    dataset = 'mnist'
    prefix = '/home/alex/mixture-gan/experiments/auto3/'
    mixtures = list(np.geomspace(0.01, 0.5, 10))
    mixtures.append(0.4)
    data = []
    for seed in range(seeds):
        for mix in mixtures:
            path = os.path.join(prefix, 'mix_%0.2f' % mix, str(seed), 'generator_model.json')
            data.append([path, mix, seed])
    columns = ['path', 'mixture', 'seed']
    return pd.DataFrame(data, columns=columns)

if __name__ == '__main__':
    runs = build_runs()
    runs.to_csv('runs.csv', index=False)
    print(runs.head())
    print(runs.shape)
