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


class TEA():
    # Implementation of Algorithm 1 of "Factored Bandits" (Zimmert and Seldin, 2018)
    def __init__(self, k, d):
        self.k = k
        self.d = d
        self.reset()

    def reset(self):
        self.t = 1
        self.T = np.array([0])
        self.TEMs = []
        for _ in range(self.d):
            self.TEMs.append(TEM(self.k))

    def update(self, observations):        
        id = np.where(self.T == self.t)[0][0]
        self.t += 1

        self.observations[id] = observations

        if self.t > self.T[-1]:
            for i in range(self.d):
                self.TEMs[i].feedback(self.observations)


    def pull_arm(self):
        if self.t > self.T[-1]:
            self.update_schedule()

        action = np.zeros(self.d, dtype=int)
        id = np.where(self.T == self.t)[0][0]
        for i in range(self.d):
            action[i] = self.TEMs[i].action_schedule[id]

        return action

    def update_schedule(self):
        values = np.zeros(self.d)

        for i in range(self.d):
            values[i] = len(self.TEMs[i].getActiveSet(1/TEA_f(self.t))) # 1 over f OR inverse of f??
        
        # M = np.argmax(values) + 1
        M = int(np.max(values)) # max OR argmax??
        self.T = np.arange(self.t, self.t + M)

        for i in range(self.d):
            self.TEMs[i].scheduleNext(M)

        self.observations = np.zeros((M))


class TEM():
    # Implementation of Algorithm 3 of "Factored Bandits" (Zimmert and Seldin, 2018)
    def __init__(self, K):
        self.K = K
        self.reset()
    
    def reset(self):
        self.N = np.zeros((self.K,self.K))
        self.D = np.zeros((self.K,self.K))
        self.B = np.arange(self.K)
        self.K_star = np.array([], dtype=int)
        self.action_schedule = np.array([], dtype=int)

    def getActiveSet(self, delta):
        if (self.N == 0).any():
            self.K_star = np.arange(self.K, dtype=int)
        else:
            self.K_star = np.array([], dtype=int)
            for i in range(self.K):
                mask = np.ones(self.K, dtype=bool)
                mask[i] = 0

                lcb = np.max((self.D[mask,i]/self.N[mask,i]) - np.sqrt((12 * np.log(2 * self.K * TEA_f(self.N[mask,i])) * 1/delta)/self.N[mask,i]))

                if lcb <= 0:
                    self.K_star = np.append(self.K_star, i)
            
            if len(self.K_star) == 0:
                self.K_star = np.arange(self.K, dtype=int)
            
            self.B = np.intersect1d(self.B, self.K_star)

            if len(self.B) == 0:
                self.B = self.K_star

        return self.K_star

    def scheduleNext(self, T):
        self.action_schedule = np.ones(T, dtype=int) * -1

        for a in self.K_star:
            t = np.random.choice(np.where(self.action_schedule == -1)[0])
            self.action_schedule[t] = a
        
        while (self.action_schedule == -1).any():
            for a in self.B:
                if (self.action_schedule != -1).all():
                    break

                t = np.random.choice(np.where(self.action_schedule != -1)[0])
                self.action_schedule[t] = a

    def feedback(self, observations):
        N = np.zeros(self.K)
        R = np.zeros(self.K)

        for t in range(len(observations)):
            R[self.action_schedule[t]] += observations[t]
            N[self.action_schedule[t]] += 1

        for i in self.K_star:
            for j in self.K_star:
                self.D[i, j] = self.D[i, j] + np.min([N[i], N[j]]) * (R[i]/N[i] - R[j]/N[j])
                self.N[i, j] = self.N[i, j] + np.min([N[i], N[j]])

def TEA_f (t):
    return (t+1) * (np.log(t+1))**2