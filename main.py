"""
main.py

Encodes the toy example models used for training in the mixture GAN
"""
from scipy.stats import norm
import numpy as np

def init_list(init_value, T):
    arr = np.zeros(T+1)
    arr[0] = init_value
    return arr


def discriminator_expectation(alpha, mu, l, r):
    discriminator_exp = 0
    for a,m in zip(alpha,mu):
        for rr in r:
            discriminator_exp+= a * norm.cdf(rr, m)
        for ll in l:
            discriminator_exp-= a * norm.cdf(ll, m)
    return discriminator_exp


def loss(alpha_star, alpha, mu_star, mu, l, r):
    """ Calculate loss function value
    L(mu, l, r) = E_{x~G*} [D(x)] + E_{x~G} [1 - D(x)] 
                            ^real            ^fake
    """
    d_real = discriminator_expectation(alpha, mu_star, l, r)
    d_fake = 1 - discriminator_expectation(alpha, mu, l, r)
    return d_real + d_fake

def opt_d_grad(alpha_star, alpha, mu_star, mu):
    """ Calculate argmax_{l,r} Loss
    
    TODO (alex): Assumes alpha_star = alpha = 0.5
    Maximizing G_star - G_t, thus intervals where G* > G^t

    There are 4! / (2*2) = 6 distinct mean orders
    """
    l = [None, None]
    r = [None, None]
    mu_all = np.concatenate([mu_star, mu])
    mo = np.argsort(mu_all)
    if mo[0] < 2: # *xxx
        if mo[1] < 2: # **tt
            pass
        else:
            if mo[2] < 2: # *t*t
                pass
            else: #*tt*
                pass
    else: # txxx
        if mo[1] < 2: # t*xx
            if mo[2] < 2: # t**t
                pass
            else: # t*t*
                pass
        else: # tt**
            pass
    return l, r


def train(alpha_star, mu_star, 
          alpha_zero, mu_zero, l_zero, r_zero, 
          step_size, T, 
          optimal_discriminator,
          train_alpha,
         ):

    # Initialize Time dependent variables
    alpha_hats = init_list(alpha_zero, T)
    mu_hats = init_list(mu_zero, T)
    l_hats = init_list(l_zero, T)
    r_hats = init_list(r_zero, T)
    for t in range(T):
        if train_alpha:
            print('ERROR: Train alpha not yet implemented')
        alpha_hats[t+1] = alpha_hats[t]

        if not optimal_discriminator:
            print('ERROR: non optimal discriminator not yet implemented')
        l_hats[t+1] = l_hats[t]
        r_hats[t+1] = r_hats[t]

        mu_hats[t+1] = mu_hats[t]


