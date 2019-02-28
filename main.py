"""
main.py

Encodes the toy example models used for training in the mixture GAN
"""
from scipy.stats import norm
import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt

def init_list(init_value, T):
    arr = [None] * T
    print(arr)
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


def F(alpha_star, alpha, mu_star, mu):
    """ function to return f(x) representing loss function values """
    def f(x):
        return (alpha_star[0] * norm.pdf(x, mu_star[0]) + 
                alpha_star[1] * norm.pdf(x, mu_star[1]) - 
                alpha[0] * norm.pdf(x, mu[0]) - 
                alpha[1] * norm.pdf(x, mu[1]))
    return f


def find_zeros(f, mu_star, mu):
    """ Finds the at most 3 zeros located in between each mean.
    Returns a list of the zeros, and which side is positive.
    If diff is positive then this root is the left end of a positive interval

    Returns zeros in sorted order

    TODO: Check that these intervals are actually the right ones to check. I.e.
    This currently assumes that the three zeros lie with at most within each mean.

    """
    intervals = np.sort(np.array(mu_star + mu)) # Sorted list concatenation
    # For numerical stability widen the range
    intervals[0] -= 1
    intervals[-1] += 1
    fun_interval_sign = np.sign(f(intervals))
    diffs = np.sign(np.diff(fun_interval_sign))  # Zero if same sign, +/-2 if different
    zeros = []
    types = []
    for i, diff in enumerate(diffs):
        if diff == 0:
            continue
        root = optimize.brentq(f, intervals[i], intervals[i+1])
        zeros.append(root)
        types.append('L' if diff > 0 else 'R')
    if len(zeros) == 0:
        print('No Zeros???')
        print('intervals', intervals)
        print('f(intervals)', f(intervals))
        print('mus', mu_star, mu)
        print('fun_interval_sign', fun_interval_sign)
    return zeros, types


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
    f = F(alpha_star, alpha, mu_star, mu)
    zeros, types = find_zeros(f, mu_star, mu)
    if len(zeros) == 0:
        plot_F(alpha_star, alpha, mu_star, mu)
        raise Exception('No zeros detected this should be analytically impossible')
    default = -10
    if len(zeros) == 1:
        if types[0] == 'R':             # + -
            l = [-np.inf, default]
            r = [zeros[0], default]
        else:                           # - +
            l = [zeros[0], default]
            r = [np.inf, default]

    if len(zeros) == 2:
        if types[0] == 'R':             # + - +
            l = [-np.inf, zeros[1]]
            r = [zeros[0], np.inf]
        else:                           # - + -
            l = [zeros[0], default]
            r = [zeros[1], default]

    if len(zeros) == 3:
        if types[0] == 'R':
            l = [-np.inf, zeros[1]]
            r = [zeros[0], zeros[2]]
        else:
            l = [zeros[0], zeros[2]]
            r = [zeros[1], np.inf]
    return l, r
def mu_grad(alpha, mu, l, r):
    """ Computes the derivatives of mu like C.1 but improved.
    """
    mu_grads = []
    for m in mu:
        v = 0
        s = 1 / np.sqrt(2 * np.pi) 
        for ll, rr in zip(l,r):
            v += norm.pdf(m, rr) - norm.pdf(m, ll)
        mu_grads.append(s*v)
    print('Mu Grad', np.array(mu_grads))
    return np.array(mu_grads)

def train(alpha_star, mu_star, 
          alpha_zero, mu_zero, l_zero, r_zero, 
          step_size, T, 
          optimal_discriminator,
          train_alpha,
         ):

    assert mu_star[0] < mu_star[1]
    assert mu_zero[0] < mu_zero[1]
    # Initialize Time dependent variables
    alpha_hats = init_list(alpha_zero, T)
    mu_hats = init_list(mu_zero, T)
    l_hats = init_list(l_zero, T)
    r_hats = init_list(r_zero, T)
    for t in range(T-1):
        if train_alpha:
            print('ERROR: Train alpha not yet implemented')
        alpha_hats[t+1] = alpha_hats[t]

        if not optimal_discriminator:
            print('ERROR: non optimal discriminator not yet implemented')
        l_hats[t+1], r_hats[t+1] = opt_d_grad(alpha_star, alpha_hats[t+1], mu_star, mu_hats[t])
        print('(l,r)', l_hats[t+1], r_hats[t+1])
        mu_hats[t+1] = mu_hats[t] - step_size * mu_grad(alpha_hats[t+1], mu_hats[t], l_hats[t+1], r_hats[t+1])
        print('Mu Hat', mu_hats[t+1])

        # Assert mu_0 < mu_1
        mu_hats[t+1] = sorted(mu_hats[t+1])
    plot_training(alpha_star, mu_star, alpha_hats, mu_hats, l_hats, r_hats, T)


def plot_training(alpha_star, mu_star, alpha_hats, 
                  mu_hats, l_hats, r_hats, T,
                  plot_intervals = True):
    fix, ax = plt.subplots(1,1)
    print(np.array(mu_hats).shape)
    mu_hats = np.array(mu_hats)
    l_hats = np.array(l_hats)
    r_hats = np.array(r_hats)
    x = np.arange(T)
    #ax.plot(x, np.repeat(mu_star[0], T), '-')
    #ax.plot(x, np.repeat(mu_star[1], T), '-')
    ax.plot(x, mu_hats[:,0], 'r', label = 'muhat0')
    ax.plot(x, mu_hats[:,1], 'g', label = 'muhat1')
    if plot_intervals:
        ax.plot(x, l_hats[:,0], '--', label = 'left0')
        ax.plot(x, l_hats[:,1], '--', label = 'left1')

        ax.plot(x, r_hats[:,0], '--', label = 'right0')
        ax.plot(x, r_hats[:,1], '--', label = 'right1')
    ax.legend()
    ax.set_ylim((-3,3))
    plt.show()

def plot_F(alpha_star, alpha, mu_star, mu):
    fix, ax = plt.subplots(1,1)
    x = np.linspace(-2, 2, 1000)
    f = F(alpha_star, alpha, mu_star, mu)
    zeros = find_zeros(f, mu_star, mu)
    print(zeros)
    #ax.plot(x, F(alpha_star, [0,0], mu_star, mu)(x), label = 'G_star')
    #ax.plot(x, F([0,0], alpha, mu_star, mu)(x), label = 'G_hat')
    ax.plot(x, F(alpha_star, alpha, mu_star, mu)(x), label = 'F')
    ax.legend(loc='best')
    plt.show()
if __name__ == '__main__':
    alpha_star = [0.5, 0.5]
    alpha_zero = [0.5, 0.5]
    mu_star = [-0.5, 0.5]
    mu_zero = [0.5,1]
    l_zero = [0,0]
    r_zero = [0,0]
    step_size = 0.1
    T = 100
    optimal_discriminator = True
    train_alpha = False
    train(alpha_star, mu_star, alpha_zero, 
          mu_zero, l_zero, r_zero, step_size, 
          T, optimal_discriminator, train_alpha)


