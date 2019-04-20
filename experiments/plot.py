from gan import GAN
import numpy as np
import click
from atongtf import util
import matplotlib.pyplot as plt
@click.command()
@click.argument('model_file', type=click.Path(exists=True))
def plot(model_file):
    def show_samples():
        noise = np.random.normal(size=(1000, 100))
        samples = model.predict(noise)
        plt.scatter(samples[:, 0], samples[:, 1], c='k')
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.savefig('%s.png' % model_file[:-5])
        plt.close()
    model = util.load_model(model_file[:-5])
    fname = model_file.split('/')[-1]
    if fname.startswith('generator'):
        show_samples()

if __name__ == '__main__':
    plot()
