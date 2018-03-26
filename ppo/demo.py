from arguments import achieve_arguments
import gym
import ppo_agent
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
    ppo_test = ppo_agent.ppo_brain(env, args)
    ppo_test.test_network()


