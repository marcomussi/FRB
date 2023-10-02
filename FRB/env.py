import numpy as np

class FactoredEnv():
    def __init__(self, n_arms_vect, dim, bounded=False, sigma=0.01,
                 min_expected=0.3, max_expected=1, seed=0, verbose=False):
        assert dim == n_arms_vect.shape[0]
        self.n_arms_vect = n_arms_vect
        self.dim = dim
        self.sigma = sigma
        self.bounded = bounded
        self.min_expected = min_expected
        self.max_expected = max_expected
        self.seed = seed
        self.verbose = verbose
        self.reset()

    def reset(self):
        self.avg_reward = []
        np.random.seed(self.seed)
        self.seed = self.seed + 1
        for size in self.n_arms_vect:
            self.avg_reward.append(
                np.random.uniform(self.min_expected, self.max_expected, size)
            )
        if self.verbose:
            print('Expected values: ' + str(self.avg_reward))
        return self

    def step(self, action):
        averages = np.zeros(self.dim)
        for i in range(self.dim):
            averages[i] = self.avg_reward[i][action[i]]
        if self.bounded:
            observations = np.zeros(self.dim)
            observations[np.random.uniform(0, 1, self.dim) < averages] = 1
        else:
            observations = averages + np.random.normal(0, self.sigma, self.dim)
        return observations

    def get_expected(self):
        return self.avg_reward
