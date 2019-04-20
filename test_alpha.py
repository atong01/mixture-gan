"""
Test the effect of varying but fixed alpha_star on the convergence rate
"""

from pprint import pprint
import pandas as pd
import numpy as np
import joblib

import main


def test_alpha_learning():
    """ Tests the convergence probability of learned alpha (always starting from
    alpha = [0.5, 0.5]) over different alpha objective parameters, over random starting
    locations.
    """
    grid_width = 10
    coords = np.linspace(-1, 1, grid_width)
    # Parameters for a given run
    runs = []
    no_of_rand_points = 10
    for alpha in np.geomspace(0.01, 0.5, 5):
        for row in range(grid_width):
            for col in range(row+1):
                for i in range(no_of_rand_points):
                    intervals = np.sort(np.random.uniform(-2, 2, size=4))
                    run = {
                        'alpha_star_1'  : alpha,
                        'alpha_star_2'  : 1 - alpha,
                        'mu_star_1'     : -0.5,
                        'mu_star_2'     : 0.5,
                        'alpha_zero_1'  : alpha,
                        'alpha_zero_2'  : 1 - alpha,
                        'mu_zero_1'     : min(coords[row], coords[col]),
                        'mu_zero_2'     : max(coords[row], coords[col]),
                        'l_zero_1'      : intervals[0],
                        'l_zero_2'      : intervals[2],
                        'r_zero_1'      : intervals[1],
                        'r_zero_2'      : intervals[3],
                        'step_size'     : 0.1,
                        'num_steps'     : 3000,
                        'optimal_discriminator' : False,
                        'train_alpha'   : True,
                        'unrolling_factor'  : 1,
                    }
                    runs.append(run)
    print('running %d runs' % len(runs))
    runs = pd.DataFrame(runs, index=range(len(runs)))
    return runs

def test_alpha_params():
    """ constructs a dataframe of parameters to run testing convergence
    probabilities over different initializations of mu
    """
    grid_width = 10
    coords = np.linspace(-1, 1, grid_width)
    # Parameters for a given run
    runs = []
    for optimal_discriminator in [True]:
    #for optimal_discriminator in [False, True]:
        no_of_rand_points = 1 if optimal_discriminator else 100
        for alpha in [0.01, 0.1, 0.25, 0.5]:
        #for alpha in np.geomspace(0.01, 0.5, 10):
            for row in range(grid_width):
                for col in range(row+1):
                    for i in range(no_of_rand_points):
                        intervals = np.sort(np.random.uniform(-2, 2, size=4))
                        run = {
                            'alpha_star_1'  : alpha,
                            'alpha_star_2'  : 1 - alpha,
                            'mu_star_1'     : -0.5,
                            'mu_star_2'     : 0.5,
                            'alpha_zero_1'  : alpha,
                            'alpha_zero_2'  : 1 - alpha,
                            'mu_zero_1'     : min(coords[row], coords[col]),
                            'mu_zero_2'     : max(coords[row], coords[col]),
                            'l_zero_1'      : intervals[0],
                            'l_zero_2'      : intervals[2],
                            'r_zero_1'      : intervals[1],
                            'r_zero_2'      : intervals[3],
                            'step_size'     : 0.1,
                            'num_steps'     : 3000,
                            'optimal_discriminator' : optimal_discriminator,
                            'train_alpha'   : False,
                            'unrolling_factor'  : 1,
                        }
                        runs.append(run)
    print('running %d runs' % len(runs))
    runs = pd.DataFrame(runs, index=range(len(runs)))
    return runs


def train_from_dict(r):
    """ Converts dict representation to argument representation """
    alpha_star = r['alpha_star_1'], r['alpha_star_2']
    mu_star = r['mu_star_1'], r['mu_star_2']
    alpha_zero = r['alpha_zero_1'], r['alpha_zero_2']
    mu_zero = r['mu_zero_1'], r['mu_zero_2']
    l_zero = r['l_zero_1'], r['l_zero_2']
    r_zero = r['r_zero_1'], r['r_zero_2']
    return main.train(alpha_star, mu_star, alpha_zero, mu_zero, l_zero, r_zero,
                      r['step_size'], r['num_steps'], 
                      r['optimal_discriminator'],
                      r['train_alpha'], r['unrolling_factor'])

def run_parallel(runs):
    p = joblib.Parallel(n_jobs=36, verbose=10)
    output = p(joblib.delayed(train_from_dict)(r) for i,r in runs.iterrows())
    return pd.DataFrame(output, columns = ['mu_t_1', 'mu_t_2'])


if __name__ == '__main__':
    runs = test_alpha_params()
    # runs = test_alpha_learning()
    df = run_parallel(runs)
    out = pd.concat([runs, df], axis=1)
    out.to_pickle('data/optimal_alpha_true.pkl')

