import numpy as np

def get_pulled_expected(vals_expected, action_vect):
    expected_pulled = 1
    for i in range(len(vals_expected)):
        expected_pulled *= vals_expected[i][action_vect[i]]
    return expected_pulled

def compute_max_expected(vals_expected):
    expected_max = 1
    for i in range(len(vals_expected)):
        expected_max *= max(vals_expected[i])
    return expected_max

def create_action_matrix(d, k):
    mx = -1 * np.ones((k**d, d),dtype=int)
    for i in range(d):
        steps = k ** (d-i-1)
        for j in range(k):
            mx[steps*j:steps*(j+1), i] = j
        for j in range(k**i):
            mx[j*k**(d-i):(j+1)*(k**(d-i)), i] = mx[0:k**(d-i), i]
    return mx

def get_sigma_square_eq_max(sigma, d, bounded):
    if bounded:
        return 0.25
    else:
        return (1 + sigma**2)**d - 1
