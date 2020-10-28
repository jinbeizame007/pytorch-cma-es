import argparse
from copy import deepcopy
import csv
import json
import os

import cma
import torch
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import gym
import ray

from models import Policy
from utils import RunningStat, compute_centered_ranks, compute_weight_decay


@ray.remote
def rollout(policy, env_name, seed=None, calc_state_stat_prob=0.01, test=False):
    save_obs = not test and np.random.random() < calc_state_stat_prob
    if save_obs:
        states = []
    env = gym.make(env_name)
    if seed is not None:
        env.seed(seed)
    state = env.reset()
    ret = 0
    timesteps = 0
    done = False
    while not done:
        if save_obs:
            states.append(state)
        with torch.no_grad():
            action = policy.act(state)
        state, reward, done, _ = env.step(action)
        ret += reward
        timesteps += 1
        if done:
            break
    if test:
        return ret
    if save_obs:
        return ret, timesteps, np.array(states)
    return ret, timesteps, None


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
parser.add_argument('--num_parallel', type=int, default=32)
parser.add_argument('--popsize', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--sigma_init', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--outdir', type=str, default='result')
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)

with open(args.outdir + "/params.json", mode="w") as f:
    json.dump(args.__dict__, f, indent=4)


def run():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gym.logger.set_level(40)
    env = gym.make(args.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    state_stat = RunningStat(
        env.observation_space.shape,
        eps=1e-2
    )
    action_space = env.action_space
    policy = Policy(state_size, action_size, args.hidden_size, action_space.low, action_space.high)
    num_params = policy.num_params
    es = cma.CMAEvolutionStrategy([0] * num_params,
                                    args.sigma_init,
                                    {'popsize': args.popsize,
                                        })
    
    ray.init(num_cpus=args.num_parallel)

    return_list = []
    for epoch in range(100000):
        #####################################
        ### Rollout and Update State Stat ###
        #####################################

        solutions = np.array(es.ask(), dtype=np.float32)
        policy.set_state_stat(state_stat.mean, state_stat.std)

        rets = []
        results = []
        for i in range(args.popsize):
            # set policy
            randomized_policy = deepcopy(policy)
            randomized_policy.set_params(solutions[i])
            # rollout
            results.append(rollout.remote(randomized_policy, args.env_name, seed=np.random.randint(0,10000000)))
        
        for result in results:
            ret, timesteps, states = ray.get(result)
            rets.append(ret)
            # update state stat
            if states is not None:
                state_stat.increment(states.sum(axis=0), np.square(states).sum(axis=0), states.shape[0])
            
        rets = np.array(rets, dtype=np.float32)
        
        best_policy_idx = np.argmax(rets)
        best_policy = deepcopy(policy)
        best_policy.set_params(solutions[best_policy_idx])
        best_rets = [rollout.remote(best_policy, args.env_name, seed=np.random.randint(0,10000000), calc_state_stat_prob=0.0, test=True) for _ in range(10)]
        best_rets = np.average(ray.get(best_rets))
        
        print('epoch:', epoch, 'mean:', np.average(rets), 'max:', np.max(rets), 'best:', best_rets)
        with open(args.outdir + '/return.csv', 'w') as f:
            return_list.append([epoch, np.max(rets), np.average(rets), best_rets])
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(return_list)

            plt.figure()
            sns.lineplot(data=np.array(return_list)[:,1:])
            plt.savefig(args.outdir + '/return.png')
            plt.close('all')
        

        #############
        ### Train ###
        #############

        ranks = compute_centered_ranks(rets)
        fitness = ranks
        if args.weight_decay > 0:
            l2_decay = compute_weight_decay(args.weight_decay, solutions)
            fitness -= l2_decay
        # convert minimize to maximize
        es.tell(solutions, fitness)


if __name__ == '__main__':
    run()