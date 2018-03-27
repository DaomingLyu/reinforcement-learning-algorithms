import gym
from dqn_agent import dqn_brain
import arguments
"""
This module was used to train the dqn network...

"""

if __name__ == '__main__':
    # achieve the arguments...
    args = arguments.achieve_args()
    # start to create the environment...
    env = gym.make(args.env)
    dqn_trainer = dqn_brain(args, env)
    # start to train the network...
    dqn_trainer.train_network()
