import tensorflow as tf
import numpy as np
import scipy.stats
import subprocess
from pprint import pprint
from joblib import Parallel, delayed
import pandas as pd
import matplotlib.pyplot as plt

from atongtf import util
import seaborn as sns


def get_samples(n, mixture):
    np.random.seed(42)
    train_data = np.random.multivariate_normal([2, 0],
                                               [[0.25, 0], [0, 0.25]],
                                               size=int((n) * mixture))
    train_data2 = np.random.multivariate_normal([-2, 0],
                                                [[0.25, 0], [0, 0.25]],
                                                size=int((n)* (1 - mixture)))
    train_data = np.concatenate([train_data, train_data2])
    return train_data

def score(model_file, alpha):
    util.set_config()
    model = util.load_model(model_file)
    z = np.random.normal(size=(100000, 100))
    y_true = get_samples(100000, alpha)
    y_pred = model.predict(z)
    wd = 0
    for i in range(2):
        wd += scipy.stats.wasserstein_distance(y_true[:,i], y_pred[:,i])**2
    wd = np.sqrt(wd)
    return wd

def score_one(model):
    iteration = int(model.split('/')[-1].split('_')[-1])
    seed = int(model.split('/')[-2])
    alpha = float(model.split('/')[-3].split('_')[-1])
    return model, iteration, seed, alpha, score(model, alpha)


def score_all(directory):
    stdout, _ = subprocess.Popen(['find', directory, '-name', 'generator_model*.json'], 
                                 stdout=subprocess.PIPE).communicate()
    models = [s.decode('utf-8')[:-5] for s in stdout.split()]
    models = [s for s in models if s.endswith('0')]
    old_scores = pd.read_pickle('scores.pkl')
    old_models = set(old_scores.path.values)
    models = [s for s in models if s not in old_models]
    scores = Parallel(n_jobs=-1, verbose=1)(delayed(score_one)(s) for s in models)
    scores = pd.DataFrame(scores, columns = ['path', 'iteration', 'seed', 'alpha', 'score'])
    scores = pd.concat([old_scores, scores])
    scores.to_pickle('scores.pkl')

from matplotlib.colors import LogNorm
if __name__ == '__main__':
    score_all('/home/alex/mixture-gan/experiments/auto/')
    scores = pd.read_pickle('scores.pkl')
    scores['alpha'] = (scores['alpha'])
    #print(scores)
    #print(np.unique(scores['alpha'], return_counts=True))
    groups = scores.groupby(by=['alpha', 'iteration'])
    #print(len(scores))
    #scores = groups.mean()
    #palette = sns.color_palette("mako_r", 9)
    palette = sns.color_palette("hls", 10)
    
    #cax = sns.lineplot(x='iteration', y='score', hue='alpha', palette=palette, data=scores, hue_norm=LogNorm(), )
    cax = sns.relplot(x='iteration', y='score', hue='alpha', palette=palette, data=scores, hue_norm=LogNorm(), kind='line', ci=None, estimator='median')
    cax.ax.set_yscale('log')
    #plt.show()
    plt.savefig('scores.png')
