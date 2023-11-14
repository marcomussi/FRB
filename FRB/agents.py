import numpy as np
from abc import ABC, abstractmethod
from random import Random
import math

class UCB1Agent():
    """
    This class implements the UCB1 algorithm in its anytime version, taken from
    (Bandits Games and Clustering Foundations, Sebastien Bubeck, PhD Thesis, 2010)
    """
    def __init__(self, n_arms, sigma, max_reward=1, exploration_alpha=4):
        self.n_arms = n_arms
        self.arms = np.arange(self.n_arms)
        self.max_reward = max_reward
        self.sigma = sigma
        self.exploration_alpha = exploration_alpha
        self.reset()

    def reset(self):
        self.t = 1
        self.last_pull = None
        self.avg_reward = np.zeros(self.n_arms)
        self.n_pulls = np.zeros(self.n_arms, dtype=int)
        return self

    def pull_arm(self):
        ucb1 = [self.avg_reward[a] + self.max_reward * self.sigma * np.sqrt(
            self.exploration_alpha * np.log(self.t) / self.n_pulls[a]) for a in range(self.n_arms)]
        self.last_pull = np.argmax(ucb1)
        return self.arms[self.last_pull]

    def update(self, reward):
        self.t += 1
        self.avg_reward[self.last_pull] = (
            self.avg_reward[self.last_pull] * self.n_pulls[self.last_pull] + reward
        ) / (self.n_pulls[self.last_pull] + 1)
        self.n_pulls[self.last_pull] += 1


class FactoredUCBAgent():
    """
    This class implements the FRB algorithm in its anytime version
    """
    def __init__(self, n_arms_vect, dim, sigma, max_reward=1, exploration_alpha=4):
        self.n_arms_vect = n_arms_vect
        self.dim = dim
        assert self.dim == self.n_arms_vect.shape[0]
        self.max_reward = max_reward
        self.sigma = sigma
        self.exploration_alpha = exploration_alpha
        self.reset()

    def reset(self):
        self.t = 1
        self.last_pull = None
        self.avg_reward = []
        self.n_pulls = []
        for size in self.n_arms_vect:
            self.avg_reward.append(np.zeros(size))
            self.n_pulls.append(np.zeros(size, dtype=int))
        return self

    def pull_arm(self):
        self.last_pull = -1 * np.ones(self.dim, dtype=int)
        for i in range(self.dim):
            ucb1 = [self.avg_reward[i][a] + self.max_reward * self.sigma * np.sqrt(
                self.exploration_alpha * np.log(self.t) / self.n_pulls[i][a]) for a in range(self.n_arms_vect[i])]
            self.last_pull[i] = int(np.argmax(ucb1))
            self.n_pulls[i][self.last_pull[i]] = self.n_pulls[i][self.last_pull[i]] + 1
        return self.last_pull

    def update(self, observations):
        self.t += 1
        for i in range(self.dim):
            self.avg_reward[i][self.last_pull[i]] = (
                self.avg_reward[i][self.last_pull[i]] *
                (self.n_pulls[i][self.last_pull[i]] - 1) + observations[i]
            ) / (self.n_pulls[i][self.last_pull[i]])


# HERE IT GOES THE FACTORED REWARD BANDITS




# Code from Gianmarco for Heavy Tails Bandits

class Agent(ABC):
    def __init__(self, n_arms, random_state=1):
        self.n_arms = n_arms
        self.random_state = random_state

    @abstractmethod
    def pull_arm(self):
        pass

    @abstractmethod
    def update(self, X, *args, **kwargs):
        pass

    def reset(self, random_state=None):
        if random_state is None:
            random_state = self.random_state
        self.t = 0
        self.last_pull = None
        self.a_hist, self.r_hist = [], []
        np.random.seed(random_state)
        self.randgen = Random(random_state)

class RobustUCBAgent(Agent):
    def __init__(self, n_arms, epsilon, u, *args, **kwargs):
        super().__init__(n_arms)
        self.epsilon, self.u = epsilon, u
        self.v, self.c = None, None
        self.reset()

    def pull_arm(self):
        ucbs = np.array([self.estimators[i]+
                        self.v*
                        (self.c/self.n_pulls[i])**(self.epsilon/(1+self.epsilon))
                        if self.n_pulls[i] > 0
                        else np.inf
                        for i in range(self.n_arms)])
        ucbs = np.nan_to_num(ucbs, nan=np.inf)
        self.last_pull = np.random.choice(np.where(ucbs == ucbs.max())[0])
        self.n_pulls[self.last_pull] += 1
        self.a_hist.append(self.last_pull)
        self.t +=1
        return self.last_pull

    def update(self, X):
        self.rewards[self.last_pull] = np.append(self.rewards[self.last_pull],X)
        self.r_hist.append(X)

    def reset(self, random_state=None):
        super().reset(random_state)
        self.rewards = [np.array([]) for i in range(self.n_arms)]
        self.estimators = np.ones(self.n_arms)*np.inf
        self.n_pulls = np.zeros(self.n_arms, dtype=int)

class TMRobustUCBAgent(RobustUCBAgent):
    def __init__(self, n_arms, epsilon, u, *args, **kwargs):
        super().__init__(n_arms, epsilon, u)
        self.v, self.c = 4*self.u**(1/(1+self.epsilon)), 0
        self.reset()

    def update(self, X):
        super().update(X)
        self.c = 2*np.log(self.t)
        for a in range(self.n_arms):
            # self.estimators[a] = np.mean(np.where(np.abs(self.rewards[a])<= self.threshold_lookup(self.n_pulls[a]),
            #                                       self.rewards[a], 0))
            self.estimators[a] = np.mean(np.where(np.abs(self.rewards[a])<= self.threshold_lookup(self.t),
                                                  self.rewards[a], 0))
            # self.estimators[a] = np.sum(np.where(np.abs(self.rewards[a]) <= self.threshold_lookup(self.t))) / self.n_pulls[a]
            # self.estimators[a] = self.trimmed_mean(self.rewards[a], self.u, self.t, self.epsilon)

    def threshold_lookup(self, n):
        # return (self.u*n*0.25/np.log(self.t))**(1/(1+self.epsilon))
        return (self.u*n/np.log(n**-2))**(1/(1+self.epsilon))
    
    def trimmed_mean(self, x, u, delta, epsilon):
        n = x.shape[0]
        mask = np.zeros(x.shape)
        _log = np.log(1/delta)
        
        t = np.arange(n)
        mask = np.abs(x) <= (u*t - _log)**(1/(1+epsilon))
        
        mask = np.array(mask, dtype='bool')

        mu = np.sum(x[mask]) / n
        return mu
