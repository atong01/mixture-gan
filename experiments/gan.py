"""
gan.py

Attempts to train a discriminator for a single univariate zero mean gaussian
dataset with the hope that the discriminator space will make sense. In that
it captures the structure of the data in a uniform way
"""


import os
from functools import partial
import itertools
import numpy as np
import keras
from keras.layers import Input, Dense
from keras.layers import LeakyReLU, Average
from keras.models import Sequential, Model
import keras.backend as K
import matplotlib.pyplot as plt
import matplotlib

from atongtf import util


class GAN():
    def __init__(self, model_dir, layer_widths=None):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(model_dir + '/data'):
            os.makedirs(model_dir + '/data')
        self.model_dir = model_dir
        self.data_dim = 2
        self.latent_dim = 100
        if layer_widths is None:
            layer_widths = [256, 128, 64]
        self.layer_widths = layer_widths
        self.build()
    
    def build(self):
        optimizer = keras.optimizers.Adam()

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim,))
        
        sample = self.generator(z)

        # Combined model only trains discriminator
        self.discriminator.trainable = False

        score = self.discriminator(sample)

        self.combined = Model(z, score)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        """ Build Generator """
        model = Sequential()
        model.add(Dense(self.layer_widths[0], input_dim=self.latent_dim))
        model.add(LeakyReLU())
        model.add(Dense(self.layer_widths[1]))
        model.add(LeakyReLU())
        model.add(Dense(self.layer_widths[2]))
        model.add(LeakyReLU())
        model.add(Dense(self.data_dim))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        sample = model(noise)

        return Model(noise, sample)

    def build_discriminator(self):
        """ Build Discriminator """
        model = Sequential()
        model.add(Dense(self.layer_widths[0], input_dim=self.data_dim))
        model.add(LeakyReLU())
        model.add(Dense(self.layer_widths[1]))
        model.add(LeakyReLU())
        model.add(Dense(self.layer_widths[2]))
        model.add(LeakyReLU())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        sample = Input(shape=(self.data_dim,))
        score = model(sample)
        return Model(sample, score)

    def train(self, num_batches, batch_size=128, sample_interval=500):
        np.random.seed(42)
        train_data = np.random.multivariate_normal([0, 0],
                                                   [[1, 0], [0, 3]],
                                                   size=(2**16))
        #self.plot_data(train_data)

        # Adversarial ground truths
        true_gt = np.ones((batch_size, 1))
        fake_gt = np.zeros((batch_size, 1))

        for batch_idx in range(num_batches):
            idx = np.random.randint(0, train_data.shape[0], batch_size)
            real = train_data[idx]
            noise = np.random.normal(size=(batch_size, self.latent_dim))
            fake = self.generator.predict(noise)

            # Train dsicriminator
            d_loss_real = self.discriminator.train_on_batch(real, true_gt)
            d_loss_fake = self.discriminator.train_on_batch(fake, fake_gt)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            # TODO Why do we want new noise??? (Alex)
            noise = np.random.normal(size=(batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, true_gt)

            print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]' %
                  (batch_idx, d_loss[0], 100*d_loss[1], g_loss))

            if batch_idx % sample_interval == 0:
                self.show_samples(batch_idx)
                #self.plot_discriminator(batch_idx)
                self.save(batch_idx)

    def plot_data(self, d):
        plt.scatter(d[:1000, 0], d[:1000, 1], c='k')
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.savefig(self.model_dir + '/train.png')
        plt.close()

    def show_samples(self, batch_idx):
        noise = np.random.normal(size=(1000, self.latent_dim))
        samples = self.generator.predict(noise)
        plt.scatter(samples[:, 0], samples[:, 1], c='k')
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.savefig(self.model_dir + '/generator_%d.png' % batch_idx)
        plt.close()

    def plot_discriminator(self, batch_idx):
        xlim = (-10, 10)
        ylim = (-10, 10)
        x = np.linspace(xlim[0], xlim[1], 100)
        y = np.linspace(ylim[0], ylim[1], 100)
        points = np.array(list(itertools.product(x, y)))
        z = self.discriminator.predict(points)

        fig, ax = plt.subplots(1, 1)
        z = z.reshape(100, 100)
        z = z.transpose()
        np.save(self.model_dir + '/data/%d.npy' % batch_idx, z)

        plt.imshow(z, cmap=matplotlib.cm.coolwarm,
                   extent=[xlim[0], xlim[1], ylim[0], ylim[1]], vmin=0,
                   vmax=1, origin='lower')
        plt.colorbar()
        plt.savefig(self.model_dir + '/discriminator_%d.png' % batch_idx)
        plt.close()

    def save(self, batch_idx=None):
        print('Saving model to: %s' % self.model_dir)
        suffix = 'model'
        if batch_idx is not None:
            suffix = 'model_%d' % batch_idx
        util.save(self.generator, self.model_dir + '/generator_%s' % suffix)
        util.save(self.discriminator, self.model_dir + '/discriminator_%s' % suffix)


class RandomWeightedAverage(Average):
    """ Provides a (random) weighted average between real and 
    generated image samples
    """
    def _merge_function(self, inputs):
        alpha = K.random_uniform((128, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class WGANGP(GAN):
    def build(self):
        self.n_critic = 5
        optimizer = keras.optimizers.RMSprop(lr=0.00005)

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # Construct Discriminator Graph
        self.generator.trainable = False
        real_sample = Input(shape=(self.data_dim,))
        z_disc = Input(shape=(self.latent_dim,))
        fake_sample = self.generator(z_disc)

        fake = self.discriminator(fake_sample)
        real = self.discriminator(real_sample)

        interpolated_sample = RandomWeightedAverage()([real_sample, 
                                                       fake_sample])
        inter = self.discriminator(interpolated_sample)

        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_sample)
        partial_gp_loss.__name__ = 'gradient_penalty'
        
        self.discriminator_model = Model(inputs=[real_sample, z_disc],
                                         outputs=[real, fake, inter])
        self.discriminator_model.compile(loss=[self.wasserstein_loss,
                                               self.wasserstein_loss,
                                               partial_gp_loss],
                                         optimizer=optimizer,
                                         loss_weights=[1, 1, 10])

        self.discriminator.trainable = False
        self.generator.trainable = True
        z_gen = Input(shape=(self.latent_dim,))
        sample = self.generator(z_gen)
        valid = self.discriminator(sample)

        self.generator_model = Model(inputs=z_gen, outputs=valid)
        self.generator_model.compile(loss=self.wasserstein_loss,
                                     optimizer=optimizer)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted 
        real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def train(self, data, num_batches, batch_size=128, sample_interval=500):
        # Adversarial ground truths
        true_gt = -np.ones((batch_size, 1))
        fake_gt = np.ones((batch_size, 1))
        dummy_gt = np.zeros((batch_size, 1))

        for batch_idx in range(num_batches+1):
            for _ in range(self.n_critic):
                idx = np.random.randint(0, data.shape[0], batch_size)
                real = data[idx]
                noise = np.random.normal(size=(batch_size, self.latent_dim))
                d_loss = self.discriminator_model.train_on_batch([real, noise],
                                                                 [true_gt, fake_gt, dummy_gt])

            # Train Generator
            # TODO Why do we want new noise??? (Alex)
            # noise = np.random.normal(size=(batch_size, self.latent_dim))
            g_loss = self.generator_model.train_on_batch(noise, true_gt)

            print('%d [D loss: %f] [G loss: %f]' %
                  (batch_idx, d_loss[0], g_loss))

            if batch_idx % sample_interval == 0:
                #self.show_samples(batch_idx)
                #self.plot_discriminator(batch_idx)
                self.save(batch_idx)

    def build_discriminator(self):
        """ Build Discriminator """
        model = Sequential()
        model.add(Dense(self.layer_widths[0], input_dim=self.data_dim))
        model.add(LeakyReLU())
        model.add(Dense(self.layer_widths[1]))
        model.add(LeakyReLU())
        model.add(Dense(self.layer_widths[2]))
        model.add(LeakyReLU())
        model.add(Dense(1))
        model.summary()

        sample = Input(shape=(self.data_dim,))
        score = model(sample)
        return Model(sample, score)


import click
@click.command()
@click.argument('path', type=click.Path())
@click.argument('mixture', type=float)
@click.argument('seed', type=int)
@click.argument('std', type=float)
def train(path, mixture, seed, std):
    util.set_config()
    model = WGANGP('%s/mix_%0.2f/%d' % (path, mixture, seed))
    np.random.seed(seed)
    train_data = np.random.multivariate_normal([2, 0],
                                               [[std, 0], [0, std]],
                                               size=int((2**10) * mixture))
    train_data2 = np.random.multivariate_normal([-2, 0],
                                                [[std, 0], [0, std]],
                                                size=int((2**10)* (1 - mixture)))
    train_data = np.concatenate([train_data, train_data2])
    np.random.shuffle(train_data)
    #model.plot_data(train_data)
    model.train(train_data, 10000, sample_interval=200)
    model.save()

if __name__ == '__main__':
    train()
