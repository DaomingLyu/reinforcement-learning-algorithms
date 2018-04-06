from arguments import achieve_arguments
import gym
import trpo_agent
import models
import mujoco_py

"""
This module was used to test the PPO algorithms...
"""

if __name__ == '__main__':
    # get the arguments
    args = achieve_arguments()
    # set up the testing environment
    env = gym.make(args.env_name)
    # start to test...
    trpo_tester = trpo_agent.trpo_brain(args, env)
    trpo_tester.test_network()

