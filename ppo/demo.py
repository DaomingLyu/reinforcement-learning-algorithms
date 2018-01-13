import gym
import ppo_agent
import models
import mujoco_py

env = gym.make('Walker2d-v1')

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

print('The number of states is ' + str(num_inputs))
print('The number of actions is ' + str(num_actions))

policy_network = models.Policy(num_inputs, num_actions)
value_network = models.Value(num_inputs)

ppo_man = ppo_agent.ppo_brain(env, policy_network, value_network, use_cuda=False)
ppo_man.test_network('saved_models/Walker2d-v1/policy_net_model_300.pt')


