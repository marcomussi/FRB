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

def lower_bound(env, T, trial):
    avg_reward = env.avg_reward[trial, :, :]
    max_val = np.max(avg_reward, axis=1).reshape(env.d, 1)
    max_idx = np.argmax(avg_reward, axis=1)
    deltas = max_val - avg_reward
    print("True deltas")
    print(deltas)
    pulls_todo = np.zeros((env.d, T), dtype=int)

    for i in range(env.d):
        pulls_todo[i, :] = max_idx[i]
        N = np.ceil(2 * (env.sigma ** 2) * np.log(T) / (deltas[i, :] ** 2)).astype(int)
        order = np.flip(np.argsort(N))
        N_ordered = N[order]
        counter = 0
        for j in range(env.k-1):
            pulls_todo[i, counter:counter + N_ordered[j]] = order[j]
            counter += N_ordered[j]
    action_vects = []
    action_vects_num = []
    expected_reward = []
    for i in range(T):
        if (i == 0):
            action_vects.append(pulls_todo[:, 0])
            action = action_vects[-1]
            exp_r = 1
            for j in range(env.d):
                exp_r *= avg_reward[j, action[j]]
            expected_reward.append(exp_r)
            action_vects_num.append(1)
        if np.array_equal(pulls_todo[:, i], action_vects[-1]):
            action_vects_num[-1] = action_vects_num[-1] + 1
        else:
            action_vects.append(pulls_todo[:, i])
            action = action_vects[-1]
            exp_r = 1
            for j in range(env.d):
                exp_r *= avg_reward[j, action[j]]
            expected_reward.append(exp_r)
            action_vects_num.append(1)

    expected_reward = np.array(expected_reward)
    action_vects_num = np.array(action_vects_num)
    optimal_reward = np.max(expected_reward)

    action_regret = optimal_reward - expected_reward

    print("Actions")
    print(action_vects)
    print("True N")
    print(action_vects_num)

    return np.dot(action_regret, action_vects_num)

# BASIC SETTING FOR EXPERIMENTS
fucb = '\\JPAalgnameshort'
ftrack = '\\stellina'
algs = [fucb, ftrack]

# k_list = [int(sys.argv[1])]
k_list = sys.argv[1].split(',')
k_list = np.array([int(i) for i in k_list])

# d_list = [int(sys.argv[2])]
d_list = sys.argv[2].split(',')
d_list = np.array([int(i) for i in d_list])

T_min = int(sys.argv[3])
T_max = int(sys.argv[4])
T_step = int(sys.argv[5])
sigma = float(sys.argv[6])
# alpha = float(sys.argv[7])
alpha_list = sys.argv[7].split(',')
alpha_list = np.array([float(i) for i in alpha_list])
parallel_workers = int(sys.argv[8])
n_trials = int(sys.argv[9])
out_folder = str(sys.argv[10])

# # DEBUG
# k_list = [2]
# d_list = [10]
# T_min = 10000
# T_max = 10000
# T_step = 1000
# sigma = 0.0
# alpha = 0
# parallel_workers = 1
# n_trials = 1
# out_folder = "./results/"

timestamp = f'{datetime.datetime.now():%Y%m%d_%H%M%S}'

out_folder = os.path.join(out_folder, timestamp)

if not os.path.exists(out_folder):
    os.makedirs(os.path.join('./', out_folder))

for d in d_list:
    for k in k_list:
        for alpha in alpha_list:
            env = ParallelFactoredEnv(k=k, d=d, num_trials=n_trials, sigma=sigma, alpha=alpha)

            T_vec = np.append(np.arange(T_min, T_max, T_step, dtype=int), T_max)
            ftrack_regret = np.zeros((n_trials, len(T_vec)))

            # regret_lbs = np.zeros((n_trials, len(T_vec)))
            # for trial in range(n_trials):
            #     for i, T in enumerate(T_vec):
            #         regret_lbs[trial, i] = lower_bound(env=env, T=T, trial=trial)

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
            for j, T in enumerate(T_vec):
                agent_ftrack = FtrackAgent(k, d, sigma, T, c=2.5)
                args = [(deepcopy(agent_ftrack), deepcopy(env), T, i) for i in range(n_trials)]
                inst_regret_ftrack = []

                with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
                    for result in executor.map(run_trial_ftrack, args):
                        inst_regret_ftrack.append(result)

                ftrack_regret[:, j] = np.sum(np.array(inst_regret_ftrack), axis=1)

            # Save data
            save_name_fucb = f"/data_fucb_T{T_max}_k{k}_d{d}_alpha{alpha}"
            save_name_ftrack = f"/data_ftrack_T{T_max}_k{k}_d{d}_alpha{alpha}"
            np.save(out_folder+save_name_fucb, fucb_regret)
            np.save(out_folder+save_name_ftrack, ftrack_regret)

            plt.figure()
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
            
            # # Lower Bounds
            # plt.plot(T_vec, np.mean(regret_lbs, axis=0), label='lower bound', marker='x')
            # plt.fill_between(T_vec,
            #                  np.mean(regret_lbs, axis=0) - np.std(regret_lbs, axis=0)/np.sqrt(n_trials),
            #                  np.mean(regret_lbs, axis=0) + np.std(regret_lbs, axis=0)/np.sqrt(n_trials),
            #                  alpha=0.3)
            
            # plt.legend()
            plt.xlabel('Rounds')
            plt.ylabel('Regret')
            plt.title('k={} d={} $\sigma$={} $alpha$={}'.format(k, d, sigma, alpha))
            save_str = out_folder + f'/ftrack_T{T_max}_k{k}_d{d}_alpha{alpha}'
            plt.savefig(save_str + '.png')
            tkz.save(save_str + '.tex')