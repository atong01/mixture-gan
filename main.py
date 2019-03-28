"""
main.py

Encodes the toy example models used for training in the mixture GAN
"""
from scipy.stats import norm
import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import seaborn as sns
import joblib
import pandas as pd
import time
from pprint import pprint


def init_list(init_value, T):
    arr = [None] * T
    arr[0] = init_value
    return arr


def discriminator_expectation(alpha, mu, l, r):
    discriminator_exp = 0
    for a, m in zip(alpha, mu):
        for rr in r:
            discriminator_exp += a * norm.cdf(rr, m)
        for ll in l:
            discriminator_exp -= a * norm.cdf(ll, m)
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
    """ function to return f(x) representing loss function values.

    Represents the difference between two univariate gaussian mixtures
    parameterized by alpha the mixture parameter and mu the means.
    alpha_star, alpha are scalar parameters where mu_star, mu are
    vectors of length 2.
    """
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
    This currently assumes that the three zeros lie with at most within each
    mean.
    """
    mu_star = list(mu_star)
    mu = list(mu)
    intervals = sorted(list(mu_star) + list(mu))  # Sorted list concatenation
    # For numerical stability add two more points
    # TODO this doesn't work in all cases and we should update it
    # to be adaptive or something smarter.
    intervals.append(intervals[-1] + 1)
    intervals.append(intervals[0] - 1)
    intervals = np.sort(np.array(intervals))

    fun_interval_sign = np.sign(f(intervals))
    # Zero if same sign, +/-2 if different
    diffs = np.sign(np.diff(fun_interval_sign))
    zeros = []
    types = []
    for i, diff in enumerate(diffs):
        if diff == 0:
            continue
        root = optimize.brentq(f, intervals[i], intervals[i+1])
        zeros.append(root)
        types.append('L' if diff > 0 else 'R')
    if len(zeros) == 0:  # This should be impossible.
        print('No Zeros??? Something went terribly wrong... Printing debug')
        print('intervals', intervals)
        print('f(intervals)', f(intervals))
        print('mus', mu_star, mu)
        print('fun_interval_sign', fun_interval_sign)
    if len(zeros) > 3:  # This should be impossible.
        print('%d Zeros??? Oh No...' % len(zeros))
    return zeros, types


def opt_d_grad(alpha_star, alpha, mu_star, mu):
    """ Calculate argmax_{l,r} Loss
    TODO (alex): Assumes alpha_star = alpha = 0.5
    Maximizing G_star - G_t, thus intervals where G* > G^t

    There are 4! / (2*2) = 6 distinct mean orders
    """
    l = [None, None]
    r = [None, None]
    f = F(alpha_star, alpha, mu_star, mu) #  f(x) = G_star - G_t at x
    zeros, types = find_zeros(f, mu_star, mu)
    if len(zeros) == 0:
        plot_F(alpha_star, alpha, mu_star, mu)
        raise Exception('No zeros detected this should be analytically impossible')
    default = -10  # default does not affect training, only plotting.
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
        if types[0] == 'R':             # + - + -
            l = [-np.inf, zeros[1]]
            r = [zeros[0], zeros[2]]
        else:                           # - + - +
            l = [zeros[0], zeros[2]]
            r = [zeros[1], np.inf]
    return l, r


def loss_gradient_mu(alpha, left, right, mu):
    """ gradient of the loss function with respect to mu"""
    grads = np.zeros(2)
    for j, m in enumerate(mu):
        grad = 0
        for l, r in zip(left, right):
            # print('pdf', l, r, norm.pdf(m, r), norm.pdf(m,l))
            grad += norm.pdf(m, r) - norm.pdf(m, l)
        grads[j] = grad
    grads *= alpha
    return grads


def loss_gradient_alpha(alpha, mu, l, r):
    """ gradient of the loss function with respect to alpha """
    grads = np.zeros(2)
    for j in [0,1]:
        for i in [0,1]:
            grads[j] += norm.cdf(r[i], mu[j]) - norm.cdf(l[i], mu[j])
    return grads

def opt_mu_grad(alpha, mu, l, r):
    """ Computes the derivatives of mu like C.1 but improved. Only works in
    the optimal discriminator case. This is a simplification of
    loss_gradient_mu function, which applies for general l, r discriminator
    bounds.
    """
    mu_grads = []
    for m in mu:
        v = 0
        s = 1 / np.sqrt(2 * np.pi) 
        for ll, rr in zip(l, r):
            v += norm.pdf(m, rr) - norm.pdf(m, ll)
        mu_grads.append(s*v)
    # print('Mu Grad', np.array(mu_grads))
    return np.array(mu_grads)


def train(alpha_star, mu_star,
          alpha_zero, mu_zero, l_zero, r_zero,
          step_size, T,
          optimal_discriminator,
          train_alpha,
          unrolling_factor):
    assert mu_star[0] <= mu_star[1]
    assert mu_zero[0] <= mu_zero[1]

    # Initialize Time dependent variables
    alpha_hats = init_list(alpha_zero, T)
    mu_hats = init_list(mu_zero, T)
    l_hats = init_list(l_zero, T)
    r_hats = init_list(r_zero, T)
    # TODO What is the best order to update these parameters?
    # Right now we have (1) alpha, (2) l,r, (3) mu
    for t in range(T-1):
        if train_alpha:
            # print('ERROR: Train alpha not yet implemented')
            agrad = loss_gradient_alpha(alpha_hats[t], mu_hats[t], l_hats[t], r_hats[t])
            alpha_hats[t+1] = alpha_hats[t] - step_size * agrad 
        else:
            alpha_hats[t+1] = alpha_hats[t]

        if optimal_discriminator:
            l_hats[t+1], r_hats[t+1] = opt_d_grad(alpha_star, alpha_hats[t+1], mu_star, mu_hats[t])
            mu_hats[t+1] = mu_hats[t] - step_size * opt_mu_grad(alpha_hats[t+1], mu_hats[t], l_hats[t+1], r_hats[t+1])
        else:  # First order dynamics
            l,r = l_hats[t],r_hats[t]
            for i in range(unrolling_factor):
                l = l - (step_size * -F(alpha_star, alpha_hats[t], mu_star, mu_hats[t])(l))
                r = r - (step_size * F(alpha_star, alpha_hats[t], mu_star, mu_hats[t])(r))
            l_hats[t+1], r_hats[t+1] = l,r

            # The following relies on the non trivial derivation by
            # wolfram alpha
            # "derivative with respect to u of integral of e^(-(x - c)^2) - 
            # e^(-(x-u)^2) with respect to x from a to b"
            # Thus is the differences of normal pdfs centered at r and those at l
            # This is a little hacky using the same function but should work
            # We changed this to r[t+1] and l[t+1]... seems better?
            mu_hats[t+1] = mu_hats[t] - step_size * loss_gradient_mu(alpha_hats[t], r_hats[t+1], l_hats[t+1], mu_hats[t])

        # print('(l,r)', l_hats[t+1], r_hats[t+1])
        # print('Mu Hat', mu_hats[t+1])
        # print('l_hats', np.array(l_hats))
        # print('r_hats', np.array(r_hats))

        # Assert mu_0 < mu_1
        mu_hats[t+1] = sorted(mu_hats[t+1])
    plot_training(alpha_star, mu_star, alpha_hats, mu_hats, l_hats, r_hats, T)
    return mu_hats[T-1]


def plot_training(alpha_star, mu_star, alpha_hats, 
                  mu_hats, l_hats, r_hats, T,
                  plot_intervals = True):
    """ Plot time series describing training behavior.
    """
    fix, ax = plt.subplots(1,1)
    print(np.array(mu_hats).shape)
    mu_hats = np.array(mu_hats)
    l_hats = np.array(l_hats)
    r_hats = np.array(r_hats)
    alpha_hats = np.array(alpha_hats)
    x = np.arange(T)
    ax.plot(x, mu_hats[:,0], 'r', label='muhat0')
    ax.plot(x, mu_hats[:,1], 'g', label='muhat1')
    ax.plot(x, alpha_hats[:,0], 'k', label='alpha0')
    if plot_intervals:
        ax.plot(x, l_hats[:,0], '--', label='left0')
        ax.plot(x, l_hats[:,1], '--', label='left1')

        ax.plot(x, r_hats[:,0], '--', label='right0')
        ax.plot(x, r_hats[:,1], '--', label='right1')
    ax.legend()
    ax.set_ylim((-3,3))
    plt.show()


def plot_F(alpha_star, alpha, mu_star, mu):
    """ Plot the TV distance as defined as G* - G_hat"""
    fix, ax = plt.subplots(1,1)
    for i in (mu_star):
        plt.axvline(i, color='k')
    for i in (mu):
        plt.axvline(i, color='b')
    x = np.linspace(-2, 2, 1000)
    # f = F(alpha_star, alpha, mu_star, mu)
    # zeros = find_zeros(f, mu_star, mu)
    # print(zeros)
    ax.plot(x, F(alpha_star, [0,0], mu_star, mu)(x), label = 'G_star')
    ax.plot(x, F([0,0], alpha, mu_star, mu)(x), label = 'G_hat')
    ax.plot(x, F(alpha_star, alpha, mu_star, mu)(x), label = 'F')
    ax.legend(loc='best')
    plt.show()


def build_heatmap_params_grid(alpha_star, mu_star, 
          alpha_zero, l_zero, r_zero, 
          step_size, T, 
          optimal_discriminator,
          train_alpha,
          unrolling_factor):
    """ returns a 2d numpy array cntaning the probabilities of convergence with different mu_0 values"""
    mu_zero = [0.0,0.0]
    grid_width = 10
    no_of_rand_points = 20         #   number of initial random points to be used for computing probabilities
    coords = np.linspace(-1, 1, grid_width)
    print(coords)
    # Parameters for a given run
    runs = []
    for row in range(grid_width):
        for col in range(row+1):
            for i in range(no_of_rand_points):
                intervals = np.sort(np.random.uniform(-3, 3, size=4))
                l_zero = [intervals[0], intervals[2]]
                r_zero = [intervals[1], intervals[3]]
                # alpha_star, mu_star
                # mu_zero[0] = random.uniform(mu[0],mu[0]+step)
                # mu_zero[1] = random.uniform(mu[1],mu[1]+step)
                mu_zero = np.sort([coords[row], coords[col]])
                params = [alpha_star, mu_star, alpha_zero, mu_zero.copy(),
                             l_zero.copy(), r_zero.copy(), step_size, T, optimal_discriminator,
                             train_alpha, unrolling_factor]
                meta = mu_zero.copy()
                runs.append((meta, params))
    print('running %d runs' % len(runs))
    return runs

def run_params_parallel(runs):
    return np.array(joblib.Parallel(n_jobs=36, verbose=10)(joblib.delayed(train)(*params) for _,params in runs))


def npdo(f, path, verbose = True):
    assert path.endswith('.npy')
    try:
        tmp = np.load(path)
        if verbose: print('Successfully loaded from file')
        return tmp
    except FileNotFoundError as inst:
        print('File not found, running given function and storing')
        start = time.time()
        out = f()
        end = time.time()
        print('Took: %d seconds.' % (end - start))
        np.save(path, out)
        return out


def generate_heatmap(runs):
    runs = build_heatmap_params_grid(alpha_star, mu_star, 
          alpha_zero, l_zero, r_zero, 
          step_size, T, 
          optimal_discriminator,
          train_alpha,
          unrolling_factor)
    mu_ts = npdo(lambda: run_params_parallel(runs), 'output.npy')
    meta = np.array(list(zip(*runs))[0])[:,::-1]
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




def train_with_early_stop(alpha_star, mu_star, 
          alpha_zero, mu_zero, l_zero, r_zero, 
          step_size, T,                                     #   T - max number of interations
          optimal_discriminator,
          train_alpha,
          unrolling_factor,
          delta                                     
         ):
    assert mu_star[0] < mu_star[1]
    assert mu_zero[0] < mu_zero[1]

    # Initialize Time dependent variables
    alpha_hats = init_list(alpha_zero, T)
    mu_hats = init_list(mu_zero, T)
    l_hats = init_list(l_zero, T)
    r_hats = init_list(r_zero, T)
    t = 0           #   number of iterations for algorithm to stop
    print (t, np.linalg.norm(mu_star-mu_hats[t]), delta)
    while(t < T-1 and np.linalg.norm(np.array(mu_star)-np.array(mu_hats[t])) > delta ):
        if train_alpha:
            print('ERROR: Train alpha not yet implemented')
        alpha_hats[t+1] = alpha_hats[t]

        if optimal_discriminator:
            l_hats[t+1], r_hats[t+1] = opt_d_grad(alpha_star, alpha_hats[t+1], mu_star, mu_hats[t])
            mu_hats[t+1] = mu_hats[t] - step_size * opt_mu_grad(alpha_hats[t+1], mu_hats[t], l_hats[t+1], r_hats[t+1])
        else:  # First order dynamics
            l,r = l_hats[t],r_hats[t]
            for i in range(unrolling_factor):
                l = l - (step_size * -F(alpha_star, alpha_hats[t], mu_star, mu_hats[t])(l))
                r = r - (step_size * F(alpha_star, alpha_hats[t], mu_star, mu_hats[t])(r))
            l_hats[t+1], r_hats[t+1] = l,r
            mu_hats[t+1] = mu_hats[t] - step_size * loss_gradient_mu(alpha_hats[t], r_hats[t+1], l_hats[t+1], mu_hats[t])

        # print('(l,r)', l_hats[t+1], r_hats[t+1])
        # print('Mu Hat', mu_hats[t+1])
        # print('l_hats', np.array(l_hats))
        # print('r_hats', np.array(r_hats))

        # Assert mu_0 < mu_1
        mu_hats[t+1] = sorted(mu_hats[t+1])
        t += 1
    
    return t


def plot_rate_of_convergence_fixed_alpha(alpha_star, mu_star, 
          l_zero, r_zero, 
          step_size,                                    
          optimal_discriminator,
          train_alpha,
          unrolling_factor,
          T,
          delta  ):
    
    no_of_points = 20          #    number of random initial points to pick for computing avg convergence time
    no_of_alpha = 10                     
    alpha = np.linspace(0.0, 1.0, no_of_alpha)

    convergence_time = np.zeros(no_of_alpha)
    convergence_prob = np.zeros(no_of_alpha)            #   fraction of points converging for each alpha
    mu_zero = np.zeros(2)
    for i in range(no_of_points):
        mu_zero[0] = random.uniform(-3,3)
        mu_zero[1] = random.uniform(-3,3) 
        mu_zero[0],mu_zero[1] = min(mu_zero),max(mu_zero)  
        for j in range(no_of_alpha):
            alpha_star[0],alpha_star[1] = alpha[j],1-alpha[j]
            alpha_zero = alpha_star
            t = train_with_early_stop(alpha_star, mu_star, alpha_zero, mu_zero, l_zero, r_zero, 
          step_size, T, optimal_discriminator, train_alpha, unrolling_factor, delta)
            if (t < T-1):          #   convergence occurs
                convergence_time[j] += t
                convergence_prob[j] += 1

    for j in range(no_of_alpha):
        if convergence_prob[j] > 0:
            convergence_time[j] = (1.0*convergence_time[j])/convergence_prob[j]
    convergence_prob = (1.0*convergence_prob)/no_of_points
    convergence_time /= (T-1)
    plt.plot(alpha,convergence_time)
    plt.plot(alpha,convergence_prob)
    plt.xlabel("alpha parameter")
    plt.legend(['convergence rate','convergence probability'])
    #plt.axis([0, 1, 0, 1])
    plt.title("Convergence rate as a function of alpha for first order dynamics")
    plt.show()






if __name__ == '__main__':

    """
    alpha_star = [0.5, 0.5]
    alpha_zero = [0.5, 0.5]
    mu_star = [-0.5, 0.5]
    mu_zero = [-10,1]
    plot_F(alpha_star, alpha_zero, mu_star, mu_zero)
    exit()
    """
    alpha_star = [0.5, 0.5]
    alpha_zero = [0.5, 0.5]
    #alpha_star = [0.5, 0.5]
    #alpha_zero = [0.5, 0.5]
    mu_star = [-0.5, 0.5]
    mu_zero = [-1,1]
    #plot_F(alpha_star, alpha_zero, mu_star, mu_zero)
    #exit()
    l_zero = [-0.75,1]
    r_zero = [0.75,0]
    step_size = 0.1
    T = 3000
    optimal_discriminator = False
    train_alpha = True
    unrolling_factor = 1
    delta = 0.1                #   tolerance
    # plot_rate_of_convergence_fixed_alpha(alpha_star, mu_star, 
    #       l_zero, r_zero, step_size,                                    
    #       optimal_discriminator, train_alpha,
    #       unrolling_factor, T, delta )

    #generate_heatmap(alpha_star, mu_star, alpha_zero, 
    #      l_zero, r_zero, step_size, 
    #      T, optimal_discriminator, train_alpha, unrolling_factor)

    train(alpha_star, mu_star, alpha_zero, 
         mu_zero, l_zero, r_zero, step_size, 
         T, optimal_discriminator, train_alpha, unrolling_factor)


