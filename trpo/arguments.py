import argparse

"""
This module was used to generate the parameters

That will be used for training the TRPO

2018-04-03

Author: Tianhong Dai

"""

def achieve_arguments():
    parse = argparse.ArgumentParser()
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor of the RL')
    parse.add_argument('--policy_lr', type=float, default=0.0005, help='the learning rate of the policy network')
    parse.add_argument('--value_lr', type=float, default=0.0005, help='the learning rate of the value network')
    parse.add_argument('--batch_size', type=int, default=64, help='the batch size to update the network')
    parse.add_argument('--env_name', default='Walker2d-v1', help='the name of the environment')
    parse.add_argument('--display_interval', type=int, default=1, help='the interval display the training information')
    parse.add_argument('--save_interval', type=int, default=100, help='the interval save the training models')
    parse.add_argument('--max_timestep', type=int, default=1000, help='the maximum time-step of each episode')
    parse.add_argument('--l2_reg', type=float, default=0.001, help='the weight decay coefficient of the optimizer')
    parse.add_argument('--max_kl', type=float, default=0.01, help='the maximal kl-divergence that allowed')
    parse.add_argument('--damping', type=float, default=0.1, help='the damping coefficient')
    parse.add_argument('--value_update_step', type=int, default=10, help='the update step for the value network')

    args = parse.parse_args()

    return args
