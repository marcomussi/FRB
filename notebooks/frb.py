# -*- coding: latin-1 -*-
import numpy as np
import math
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tikzplotlib as tkz
import warnings
warnings.filterwarnings("ignore")

import os, sys

_, filename = os.path.split(os.getcwd())
if filename == 'notebooks':
    old_dir = os.getcwd()
    os.chdir('../')
    print('Moving Current Directory from ' + old_dir + ' to ' + os.getcwd())
else:
    print('Current Directory is ' + os.getcwd())

sys.path.append('./')

from FRB.agents import UCB1Agent, FactoredUCBAgent, TEA
from FRB.env import FactoredEnv
from FRB.utils import get_pulled_expected, compute_max_expected, create_action_matrix, get_sigma_square_eq_max


class TMRobustUCBAgent:
    
    def __init__(self, n_arms, u, mult=1, *args, **kwargs):
        self.n_arms = n_arms
        self.u = u
        self.mult = mult
        self.v = np.sqrt(self.u)
        self.e_sqrt_16 = np.exp(1/16)
        self.reset()
        
    def pull_arm(self):
        ucbs = self.estimators + self.mult * np.sqrt(12 * self.v * 32 * np.log(self.e_sqrt_16 * self.t) / self.n_pulls)
        ucbs = np.nan_to_num(ucbs, nan=np.inf)
        self.last_pull = np.random.choice(np.where(ucbs == ucbs.max())[0])
        self.n_pulls[self.last_pull] += 1
        self.t += 1
        return self.last_pull
    
    def update(self, X):
        self.rewards[self.last_pull] = np.append(self.rewards[self.last_pull], X)
        self.c = 2+32*np.log(self.t)
        for a in range(self.n_arms):
            k = max(int(min(self.c, self.n_pulls[a])/2), 1)
            N = int(self.n_pulls[a]/k)
            self.estimators[a] = np.median([np.mean(chunk) for chunk in np.array_split(self.rewards[a][:N*k], k)])

    
    def threshold_lookup(self, n):
        return np.sqrt(self.u * n / (-2 * np.log(self.t)))
    
    def reset(self):
        self.t = 1
        self.last_pull = None
        self.rewards = [np.array([]) for i in range(self.n_arms)]
        self.estimators = np.ones(self.n_arms)*np.inf
        self.n_pulls = np.zeros(self.n_arms, dtype=int)


class NewFactoredUCBAgentMM():
    """
    This class implements the FRB MM optimal algorithm in its anytime
    version for bounded variables
    """
    def __init__(self, k, d, T, bounded, sigma):
        self.k = k
        self.d = d
        self.T = T
        if bounded:
            self.v = sigma
        else:
            self.u = (1 + sigma**2)**d - 1
            self.v = np.sqrt(self.u)
        self.num_actions = self.k ** self.d
        self.bounded = bounded
        # Creation of the action matrix
        self.action_matrix = np.zeros(
            (self.num_actions, self.d), dtype=int
        )
        for i in range(self.d):
            vect = -1 * np.ones(self.k**(i+1))
            external_repeats = int(self.k**(self.d-(i+1)))
            internal_repeats = self.k**i
            for j in range(self.k):
                vect[j*internal_repeats:(j+1)*internal_repeats] = j
            vect_new = np.copy(vect).reshape(-1, 1)
            for _ in range(external_repeats-1):
                vect_new = np.vstack((vect_new, vect.reshape(-1, 1)))
            self.action_matrix[:, i] = vect_new.ravel()
        self.e_sqrt_16 = np.exp(1/16)
        self.reset()

    def update(self, observations):
        self.t += 1
        for i in range(self.d):
            self.observations[i, self.last_pull[i], self.n_pulls[i, self.last_pull[i]]] = observations[i]
            self.n_pulls[i, self.last_pull[i]] = self.n_pulls[i, self.last_pull[i]] + 1

    def threshold_lookup(self, n):
        return np.sqrt(self.u * n / (-2 * np.log(self.t)))
    
    def pull_arm(self):
        if self.bounded:
            for i in range(self.num_actions):
                action_vector = self.action_matrix[i, :]
                new_min_pull = self.n_pulls[0, action_vector[0]]
                for j in range(1, self.d):
                    new_min_pull = min(new_min_pull, self.n_pulls[j, action_vector[j]])
                if new_min_pull != self.n_min_pull[i]:
                    self.n_min_pull[i] = new_min_pull
                    aux = 1
                    for j in range(self.d):
                        aux *= self.observations[j, self.last_pull[j], self.n_min_pull[i]-1]
                    self.virtual_pulls_sum[i] += aux
            mean = self.virtual_pulls_sum / self.n_min_pull
            ucb = mean + self.v * np.sqrt(4 * math.log(self.t) / self.n_min_pull)
        else:
        
            for i in range(self.num_actions):
                
                action_vector = self.action_matrix[i, :]
                if (action_vector == self.last_pull).any():
                    new_min_pull = self.n_pulls[0, action_vector[0]]
                    for j in range(1, self.d):
                        new_min_pull = min(new_min_pull, self.n_pulls[j, action_vector[j]])
                    self.n_min_pull[i] = new_min_pull

                _observations = np.zeros((self.d, self.n_min_pull[i]))

                for h in range(self.d):
                    _observations[h, :] = self.observations[h, action_vector[h], :self.n_min_pull[i]]

                x = np.prod(_observations, axis=0)
                
                k = max(int(min(2+32*np.log(self.t), self.n_min_pull[i])/2), 1)
                N = int(self.n_min_pull[i]/k)
                self.virtual_pulls_sum[i] = np.median([np.mean(chunk) for chunk in np.array_split(x[:N*k], k)])

            mean = self.virtual_pulls_sum
            
            ucb = mean + np.sqrt(12 * self.v * 32 * np.log(self.e_sqrt_16 * self.t) / self.n_min_pull)
            ucb = np.nan_to_num(ucb, nan=np.inf)
        
        self.last_pull = self.action_matrix[np.random.choice(np.where(ucb == ucb.max())[0]), :]

        return self.last_pull
                       
    def reset(self):
        self.t = 1
        self.last_pull = None
        self.n_min_pull = np.zeros(self.num_actions, dtype=int)
        self.n_pulls = np.zeros((self.d, self.k), dtype=int)
        self.observations = -1 * np.ones((self.d, self.k, self.T))
        self.virtual_pulls_sum = np.zeros(self.num_actions)

# Body of the script

# BASIC SETTING FOR EXPERIMENTS
fucb = '\\JPAalgnameshort'
fucbMM = '\\JPAalgnameshortMM'
ucbone = '\\ucbone'
httem = '\\httem'
tea = '\\tea'
algs = [fucb, ucbone, httem]# , tea]
# T = 100000
checkpoints = [1000, 5000, 10000]
n_trials = 50
seed = 0
# k_list = [3, 5]
# d_list = [1, 2, 3, 4]
k_list = [int(sys.argv[1])]
d_list = [int(sys.argv[2])]
T = int(sys.argv[3])
bounded_list = [False]
do_subsampling = True

# OVERRIDE FOR TESTING PURPOSE TO SPEED UP THE RUNS
# T = 10000
# checkpoints = [1000, 2000, 5000]
# bounded_list = [False]
# algs = [fucb, tea]
# n_trials = 4
# k_list = [3]
# d_list = [1]
# do_subsampling = True
    
result_table = {}
out_folder = str(sys.argv[4])

ht_mult = float(sys.argv[5])
_sigma = float(sys.argv[6])

for bounded in bounded_list:

    result_table[bounded] = {}
    
    if bounded: 
        sigma = 0.5 # fixed for bernoulli
    else:
        sigma = _sigma
    
    for d in d_list:

        result_table[bounded][d] = {}

        for k in k_list:

            out_path = out_folder + 'out' + str(k) + '_' + str(d) + '.txt'

            result_table[bounded][d][k] = {}

            arms_vect = k * np.ones(d, dtype=int)

            # F-UCB INIT
            agent_factored = FactoredUCBAgent(arms_vect, d, sigma)

            # F-UCB-MM INIT
            agent_factored_MM = NewFactoredUCBAgentMM(k, d, T, bounded, sigma)
            
            # UCB1 INIT
            agent_ucb = UCB1Agent(d*k, sigma)
            action_mx = create_action_matrix(d, k)

            # HT-TEM INIT
            agent_ht_tem = TMRobustUCBAgent(n_arms=d*k, u=(1+sigma**2)**d-1, mult=ht_mult)

            # TEA INIT
            agent_tea = TEA(k, d)
            
            mean_cum_expected_regret = {}
            std_cum_expected_regret = {}
            
            for alg in algs:

                result_table[bounded][d][k][alg] = {}

                env = FactoredEnv(arms_vect, d, sigma=sigma, bounded=bounded)

                inst_expected_regret = np.zeros((n_trials, T))
                
                # for trial_i in range(n_trials):
                for trial_i in tqdm(range(n_trials)):
                
                    vals_expected = env.get_expected()
                    max_expected = compute_max_expected(vals_expected)

                    for t in range(T):

                        if alg == ucbone:
                            action = action_mx[agent_ucb.pull_arm(), :]
                            agent_ucb.update(np.prod(env.step(action)))
                        elif alg == fucb:
                            action = agent_factored.pull_arm()
                            agent_factored.update(env.step(action))
                        elif alg == fucbMM:
                            action = agent_factored_MM.pull_arm()
                            agent_factored_MM.update(env.step(action))
                        elif alg == httem:
                            action = action_mx[agent_ht_tem.pull_arm(), :]
                            agent_ht_tem.update(np.prod(env.step(action)))
                        elif alg == tea:
                            action = agent_tea.pull_arm()
                            agent_tea.update(np.prod(env.step(action)))
                        else:
                            raise ValueError('Error in selecting algorithm')

                        inst_expected_regret[trial_i, t] = max_expected - get_pulled_expected(
                            vals_expected, action)
                    
                    # I reset all the agents, becuase i do not know which one 
                    # i am using for the sake of simplicity
                    
                    if trial_i < n_trials - 1:
                        env.reset()
                        agent_ucb.reset()
                    agent_factored.reset()
                    agent_factored_MM.reset()
                    agent_ht_tem.reset()
                    agent_tea.reset()
                
                # maybe replace with cumsum with correct axis
                cum_expected_regret = np.zeros(inst_expected_regret.shape)
                cum_expected_regret[:, 0] = inst_expected_regret[:, 0]
                for i in range(1, T):
                    cum_expected_regret[:, i] = inst_expected_regret[:, i] + cum_expected_regret[:, i-1]

                mean_cum_expected_regret[alg] = np.mean(cum_expected_regret, axis=0)
                std_cum_expected_regret[alg] = np.std(cum_expected_regret, axis=0) / np.sqrt(n_trials)

                print('{} run completed - k={} d={} $\sigma$={}'.format(alg, k, d, sigma))
                for i in checkpoints:
                    result_table[bounded][d][k][alg][i] = '${} \ ({})$   '.format(
                        round(mean_cum_expected_regret[alg][i-1], 2), 
                        round(std_cum_expected_regret[alg][i-1], 2)
                    )
                    print('T={}: ${} \ ({})$'.format(i, round(mean_cum_expected_regret[alg][i-1], 2), 
                                                  round(std_cum_expected_regret[alg][i-1], 2)))

            plt.figure()
            if do_subsampling:
                subsample = 50
                assert T % subsample == 0
                x_plt = np.linspace(0, T-1, int(T/subsample), dtype=int)
            else:
                x_plt = np.linspace(0, T-1, T, dtype=int)
            for alg in algs:
                plt.plot(x_plt, mean_cum_expected_regret[alg][x_plt], 
                         label=alg)
                plt.fill_between(x_plt, 
                                 mean_cum_expected_regret[alg][x_plt] - std_cum_expected_regret[alg][x_plt], 
                                 mean_cum_expected_regret[alg][x_plt] + std_cum_expected_regret[alg][x_plt], 
                                 alpha=0.3)
            plt.legend()
            plt.xlabel('Rounds')
            plt.ylabel('Regret')
            plt.title('bounded={} k={} d={} $\sigma$={}'.format(bounded, k, d, sigma))
            if bounded:
                save_str = out_folder + 'bounded_k{}_d{}'.format(k, d)
            else:
                save_str = out_folder + 'subgauss_k{}_d{}'.format(k, d)
            plt.savefig(save_str + '.png')
            tkz.save(save_str + '.tex')


with open(out_path, 'w') as f:
    f.write('d= \t k= \t T=\t\t')
    [f.write(alg + '\t\t\t') for alg in algs]
    f.write('\n')

    for d in d_list:
        for k in k_list:
            for T_val in checkpoints:
                f.write('${}$ & \t ${}$ & \t ${}$ \t\t'.format(d, k, T_val))
                for bounded in bounded_list:
                    for alg in algs:
                        f.write('&' + str(result_table[bounded][d][k][alg][T_val]) + '\t')
                if T_val == checkpoints[-1]:
                    f.write('\\\\\n\\cmidrule{2-10}')
                else:
                    f.write('\\\\\n\\cmidrule{3-10}')
        f.write('\cmidrule{1-10}')
