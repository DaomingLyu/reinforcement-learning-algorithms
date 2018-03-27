import gym
from dqn_agent import dqn_brain
import arguments

"""
This module was used to run the demo of the DQN...

"""

if __name__ == '__main__':
    # achieve the arguments...
    args = arguments.achieve_args()

    # start to create the environment...
    env = gym.make(args.env)
    # start to define the class..
    dqn_tester = dqn_brain(args, env)
    dqn_tester.test_network()
