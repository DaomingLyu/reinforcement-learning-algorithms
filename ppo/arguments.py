import argparse
"""
This module was used to setup the arguments that

will be used for training the agent and test the agent

2018-03-24

"""

# define the function to get the arguments...
def achieve_arguments():
    parse = argparse.ArgumentParser()
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount value of the RL')
    parse.add_argument('--policy_lr', type=float, default=0.0005, help='the learning rate of the actor network')
    parse.add_argument('--value_lr', type=float, default=0.0005, help='the learning rate of the critic network')
    parse.add_argument('--tau', type=float, default=0.95, help='the coefficient to calculate the GAE')
    parse.add_argument('--epsilon', type=float, default=0.2, help='the coefficient to clip the surrogate function')
    parse.add_argument('--policy_update_step', type=int, default=10, help='the update step of the policy network')
    parse.add_argument('--value_update_step', type=int, default=10, help='the update step of the critic network')
    parse.add_argument('--batch_size', type=int, default=64, help='the batch size of the ppo')
    parse.add_argument('--episode_length', type=int, default=200, help='the maximum length per episode...')
    parse.add_argument('--cuda', action='store_true', help='use the cuda to train the agent')
    parse.add_argument('--env_name', default='Walker2d-v1', help='the name of the environment')
    parse.add_argument('--display_interval', type=int, default=1, help='the interval display the training information')
    parse.add_argument('--save_interval', type=int, default=100, help='the interval save the training models')
    parse.add_argument('--max_timestep', type=int, default=1000, help='the maximum time-step of each episode')
    parse.add_argument('--l2_reg', type=float, default=0.001, help='the weight decay coefficient of the optimizer')

    args = parse.parse_args()

    return args 

