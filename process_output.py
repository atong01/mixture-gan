import pandas as pd
import numpy as np
def generate_heatmap(fname):
    df = pd.read_pickle(fname)
    print(df)
    tol = 0.1
    converged = np.all(np.abs(mu_ts - mu_star) < tol, axis=1).astype(np.float)

    meta_df = pd.DataFrame(meta, columns = ['mu_0', 'mu_1'])
    output = pd.DataFrame(converged, columns = ['Converged'])
    df = pd.concat([meta_df, output], axis=1)
    print(df)
    df_agg = df.groupby(['mu_0', 'mu_1'], as_index=False).agg('mean')
    sns.heatmap(df_agg.pivot('mu_0', 'mu_1', 'Converged'), 
                vmin=0, 
                # vmax=1, 
                cmap='jet')
    plt.show()

if __name__ == '__main__':
    fname = 'output.pkl'
    generate_heatmap(fname)
