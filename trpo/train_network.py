import gym
import trpo_agent
import mujoco_py
from arguments import achieve_arguments

if __name__ == '__main__':
    args = achieve_arguments()
    # build up the training environment
    env = gym.make(args.env_name)
    # start to train the environment
    trpo_trainer = trpo_agent.trpo_brain(args, env)
    # train the network...
    trpo_trainer.train_network()


