import numpy as np
import torch
import multiprocessing
import threading
import models
import gym
import torch.multiprocessing as mp
from a3c_agent_discrete import A3C_Workers

torch.set_default_tensor_type('torch.DoubleTensor')

if __name__ == '__main__':
	env_name = 'Pong-v0'

	save_path = 'saved_models/Pong-v0/'
	# the number of cpu...
	num_of_workers = multiprocessing.cpu_count()
	# build up the center network....
	actor_critic_global = models.Actor_Critic(2)
	actor_critic_global.share_memory()

	print(actor_critic_global)

    # build up the workers...
	workers = []
	processor = []
	
	for idx in range(num_of_workers):
		if idx == 0:
			workers.append(A3C_Workers(env_name, info=True))
		else:
			workers.append(A3C_Workers(env_name))

	# add them into the multiprocessor....
	for worker in workers:
		process = mp.Process(target=worker.train_network, args=(actor_critic_global, save_path))
		process.start()
		processor.append(process)

	for p in processor:
		p.join()

