import main
from scipy.stats import norm

def test_discriminator_expectation():
    # Test discriminator_expectation
    alpha = [0.5, 0.5]
    mu = [0,50]
    l = [-1, 49]
    r = [1, 51]
    answer = main.discriminator_expectation(alpha, mu, l, r)
    # area within 1 std
    correct = 0.682689492137
    assert abs(answer - correct) < 1e-4

def test_discriminator_expectation_alpha():
    # Test alpha
    alpha = [0.9, 0.1]
    mu = [0,50]
    l = [-1, 49]
    r = [-1, 51]
    answer = main.discriminator_expectation(alpha, mu, l, r)
    correct = 0.1 * 0.682689492137
    assert abs(answer - correct) < 1e-4

if __name__ == '__main__':
    test_discriminator_expectation()
    test_discriminator_expectation_alpha()
