"""
Takes the output of an output dataframe and converts it to a set of heatmaps

TODO: Expand to more general dataframes than the alpha_exploration
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_heatmap(fname):
    """ From a file containing a pickled output dataframe produces a set of heatmaps
    """
    dataframe = pd.read_pickle(fname)
    mu_ts = np.array(dataframe[['mu_t_1', 'mu_t_2']])
    mu_star = np.array(dataframe[['mu_star_1', 'mu_star_2']])
    tol = 0.1
    converged = np.all(np.abs(mu_ts - mu_star) < tol, axis=1).astype(np.float)
    output = pd.DataFrame(converged, columns=['Converged'])
    df = pd.concat([dataframe, output], axis=1)
    for optimal_discriminator in [True]:
    #for optimal_discriminator in [False, True]:
        for alpha in sorted(df['alpha_star_1'].unique()):
            mask = np.bitwise_and(df.alpha_star_1 == alpha,
                                  df.optimal_discriminator == optimal_discriminator)
            df_agg = df[mask].groupby(['mu_zero_1', 'mu_zero_2'], as_index=False)
            df_agg = df_agg.agg({'Converged' : 'mean'})
            df_agg = df_agg.round(3)
            sns.heatmap(df_agg.pivot('mu_zero_1', 'mu_zero_2', 'Converged'),
                        vmin=0,
                        vmax=1,
                        cmap='jet')
            title = 'alpha_%0.3f_opt_%s.png' % (alpha, optimal_discriminator)
            plt.title(title)
            plt.tight_layout()
            plt.savefig('images/true/%s' % title)
            plt.close()


if __name__ == '__main__':
    FNAME = 'data/optimal_discriminator_alpha_exploration.pkl'
    FNAME = 'data/optimal_alpha_true.pkl'
    #FNAME = 'data/alpha_exploration.pkl'
    generate_heatmap(FNAME)
