import numpy as np
import torch
import multiprocessing
import threading
import models
import gym
import torch.multiprocessing as mp
from a3c_agent_continues import A3C_Workers

torch.set_default_tensor_type('torch.DoubleTensor')

if __name__ == '__main__':
	env_name = 'Pendulum-v0'
	save_path = 'saved_models/Pendulum-v0/'
	# the number of cpu...
	num_of_workers = multiprocessing.cpu_count()

	env = gym.make(env_name)
	num_inputs = env.observation_space.shape[0]
	num_actions = env.action_space.shape[0]

	# build up the center network....
	value_network_global = models.Value(num_inputs)
	policy_network_global = models.Policy(num_inputs, num_actions)

	value_network_global.share_memory()
	policy_network_global.share_memory()

	# build up the workers...
	workers = []
	processor = []

	#worker_test = A3C_Workers(env_name)
	#worker_test.test_the_network(path='saved_models/policy_model_3700.pt')
	for idx in range(num_of_workers):
		if idx == 0:
			workers.append(A3C_Workers(env_name, True))
		else:
			workers.append(A3C_Workers(env_name))

	# add them into the multiprocessor....
	for worker in workers:
		process = mp.Process(target=worker.train_network, args=(value_network_global, policy_network_global, save_path))
		process.start()
		processor.append(process)

	for p in processor:
		p.join()

