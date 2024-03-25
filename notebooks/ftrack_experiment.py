import numpy as np
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tikzplotlib as tkz
import os, sys
import time
import datetime
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

_, filename = os.path.split(os.getcwd())
if filename == 'notebooks':
    old_dir = os.getcwd()
    os.chdir('../')
    print('Moving Current Directory from ' + old_dir + ' to ' + os.getcwd())
else:
    print('Current Directory is ' + os.getcwd())

sys.path.append('./')

from FRB.agents import FactoredUCBAgent, FtrackAgent
from FRB.env import ParallelFactoredEnv
from FRB.utils import get_pulled_expected, compute_max_expected

def run_trial_fucb(arg):
    start = time.time()
    agent = arg[0]
    env = arg[1]
    T = arg[2]
    trial = arg[3]

    print('F-UCB: Started run {}.\n'.format(trial), end='')

    vals_expected = list(env.get_expected(trial))
    max_expected = compute_max_expected(vals_expected)
    result = np.zeros(T, dtype=float)

    for i in range(T):
        action = agent.pull_arm()
        agent.update(env.step(trial, action))

        result[i] = max_expected - get_pulled_expected(vals_expected, action)

    end = time.time()
    print('F-UCB: Ended run {}. Elapsed time: {:.2f}s.\n'.format(trial, end-start), end='')
    return result

def run_trial_ftrack(arg):
    start = time.time()
    agent = arg[0]
    env = arg[1]
    T = arg[2]
    trial = arg[3]

    print('F-Track: Started run {} (T={}).\n'.format(trial, T), end='')

    vals_expected = list(env.get_expected(trial))
    max_expected = compute_max_expected(vals_expected)
    result = np.zeros(T, dtype=float)

    for i in range(T):
        action = agent.pull_arm()
        agent.update(env.step(trial, action))

        result[i] = max_expected - get_pulled_expected(vals_expected, action)

    end = time.time()
    print('F-Track: Ended run {} (T={}). Elapsed time: {:.2f}s.\n'.format(trial, T, end-start), end='')
    return result

# BASIC SETTING FOR EXPERIMENTS
fucb = '\\JPAalgnameshort'
ftrack = '\\stellina'
algs = [fucb, ftrack]

k_list = [int(sys.argv[1])]
d_list = [int(sys.argv[2])]
T_min = int(sys.argv[3])
T_max = int(sys.argv[4])
T_step = int(sys.argv[5])
sigma = float(sys.argv[6])
parallel_workers = int(sys.argv[7])
n_trials = int(sys.argv[8])
out_folder = str(sys.argv[9])

if not os.path.exists(out_folder):
    os.makedirs(os.path.join('./', out_folder))

for d in d_list:
    for k in k_list:
        env = ParallelFactoredEnv(k, d, n_trials, sigma)

        # F-UCB
        arms_vect = k * np.ones(d, dtype=int)
        agent_fucb = FactoredUCBAgent(arms_vect, d, sigma)

        args = [(deepcopy(agent_fucb), deepcopy(env), T_max, i) for i in range(n_trials)]
        inst_regret_fucb = []

        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            for result in executor.map(run_trial_fucb, args):
                inst_regret_fucb.append(result)

        fucb_regret = np.array(inst_regret_fucb)
        fucb_regret = np.cumsum(fucb_regret, axis=1)

        # F-Track
        T_vec = np.append(np.arange(T_min, T_max, T_step, dtype=int), T_max)
        ftrack_regret = np.zeros((n_trials, len(T_vec)))

        for j, T in enumerate(T_vec):
            agent_ftrack = FtrackAgent(k, d, sigma, T, c=2.5)
            args = [(deepcopy(agent_ftrack), deepcopy(env), T, i) for i in range(n_trials)]
            inst_regret_ftrack = []

            with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
                for result in executor.map(run_trial_ftrack, args):
                    inst_regret_ftrack.append(result)

            ftrack_regret[:, j] = np.sum(np.array(inst_regret_ftrack), axis=1)

        plt.figure()
        subsample = 50
        assert T % subsample == 0
        # F-UCB
        plt.plot(T_vec, np.mean(fucb_regret, axis=0)[T_vec-1], label=fucb, marker='x')
        plt.fill_between(T_vec,
                         np.mean(fucb_regret, axis=0)[T_vec-1] - np.std(fucb_regret, axis=0)[T_vec-1]/np.sqrt(n_trials),
                         np.mean(fucb_regret, axis=0)[T_vec-1] + np.std(fucb_regret, axis=0)[T_vec-1]/np.sqrt(n_trials),
                         alpha=0.3)
        
        # F-Track
        plt.plot(T_vec, np.mean(ftrack_regret, axis=0), label=ftrack, marker='x')
        plt.fill_between(T_vec,
                         np.mean(ftrack_regret, axis=0) - np.std(ftrack_regret, axis=0)/np.sqrt(n_trials),
                         np.mean(ftrack_regret, axis=0) + np.std(ftrack_regret, axis=0)/np.sqrt(n_trials),
                         alpha=0.3)
        
        plt.legend()
        plt.xlabel('Rounds')
        plt.ylabel('Regret')
        plt.title('k={} d={} $\sigma$={}'.format(k, d, sigma))
        save_str = out_folder + f'ftrack_T{T_max}_k{k}_d{d}_{datetime.datetime.now():%Y-%m-%d_%H%M%S}'
        plt.savefig(save_str + '.png')
        tkz.save(save_str + '.tex')