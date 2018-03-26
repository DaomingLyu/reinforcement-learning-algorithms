import gym
import ppo_agent
import mujoco_py
from arguments import achieve_arguments

if __name__ == '__main__':
    args = achieve_arguments()
    # build up the training environment
    env = gym.make(args.env_name)
    # start to train the environment
    ppo_trainer = ppo_agent.ppo_brain(env, args)
    ppo_trainer.train_network()




