import numpy as np
import torch
import multiprocessing
import threading
import models
import gym
import torch.multiprocessing as mp
from torch.autograd import Variable
import torch.optim as optim

torch.set_default_tensor_type('torch.DoubleTensor')

# build up the A3C Workers....
class A3C_Workers():
	def __init__(self, env_name, info=False, gamma=0.9, entropy_beta=0.01, global_update_step=20):
		self.env = gym.make(env_name)
		num_inputs = self.env.observation_space.shape[0]
		num_actions = self.env.action_space.shape[0]

		# init some other parameters....
		self.gamma = gamma
		self.global_update_step = global_update_step
		self.info = info

		# build up the personal network...
		self.value_network_local = models.Value(num_inputs)
		self.policy_network_local = models.Policy(num_inputs, num_actions)

	# train the network....
	def train_network(self, value_network_global, policy_network_global, path):
		# define as training mode...
		self.value_network_local.train()
		self.policy_network_local.train()

		# define the optimizer...
		value_optimizer = optim.RMSprop(value_network_global.parameters(), lr=0.001)
		policy_optimizer = optim.RMSprop(policy_network_global.parameters(), lr=0.0001)
		# init the thread time step...
		t = 1
		eposide_num = 0
		reward_sum = 0
		state = self.env.reset()
		while True:
			# load the global network's parametesrs...
			self.value_network_local.load_state_dict(value_network_global.state_dict())
			self.policy_network_local.load_state_dict(policy_network_global.state_dict())
			t_start = t
			terminal = False
			brain_memory = []
			while terminal == False and t - t_start < self.global_update_step:
				# convert it into tensor...
				state_tensor = torch.from_numpy(state).view(1, -1)
				# input the state into the policy networl...
				action, log_p = self.policy_network_local(Variable(state_tensor))
				# select the action...
				action = action.data.numpy()[0]
				# because the beta distribution is from 0 ~ 1.... but the bound for Reacher-v1 [-1, 1]
				# you could search the infomation through env.action_space.low and env.action_space.high...
				action = -2 + action * 4
				# conduct the action...
				state_, reward, terminal, _ = self.env.step(action)
				# add the reward...
				reward_sum += reward
				t += 1
				# store the useful informaiton....
				brain_memory.append((state, log_p, reward, terminal))

				state = state_

			if terminal == False:
				state_tensor = torch.from_numpy(state).view(1, -1)
				previous_return = self.value_network_local(Variable(state_tensor))
				previous_return = previous_return.squeeze(0)
				previous_return = previous_return.detach()

			else:
				previous_return = 0
			# start to udate the network....
			v_loss, p_loss = self.update_the_global_network(brain_memory, value_network_global, policy_network_global, value_optimizer, policy_optimizer, previous_return)
			
			if terminal:
				state = self.env.reset()
				if self.info:
					if eposide_num % 10 == 0:
						print('The eposide_num is ' + str(eposide_num) + ', and the reward_sum is ' + str(reward_sum) + 
							', the value loss is: ' + str(v_loss) + ', and the policy loss is: ' + str(p_loss)) 
					if eposide_num % 100 == 0:
						save_path = path + 'policy_model_' + str(eposide_num) + '.pt'
						torch.save(policy_network_global.state_dict(), save_path)
						print('----------------------------------------------------')
						print('model has been saved!')
						print('----------------------------------------------------')
				reward_sum = 0
				eposide_num += 1

	# this is used to update the global network...
	def update_the_global_network(self, brain_memory, value_network_global, policy_network_global, value_optimizer, policy_optimizer, previous_return):
		state_batch = np.array([element[0] for element in brain_memory])
		state_batch_tensor = torch.from_numpy(state_batch)

		log_p_batch = [element[1] for element in brain_memory]

		reward_batch = np.array([element[2] for element in brain_memory])
		reward_batch_tensor = torch.from_numpy(reward_batch)
		reward_batch_tensor = Variable(reward_batch_tensor)

		terminal_batch = [element[3] for element in brain_memory]

		predicted_value = self.value_network_local(Variable(state_batch_tensor))

		# get the returns and the advantages....
		value_loss, policy_loss = self.calculate_the_loss(reward_batch_tensor, predicted_value, terminal_batch, log_p_batch, previous_return)

		# start to update...
		value_optimizer.zero_grad()
		policy_optimizer.zero_grad()

		value_loss.backward()
		policy_loss.backward()

		self.transfer_grads_to_shared_models(self.value_network_local, value_network_global, self.policy_network_local, policy_network_global)

		value_optimizer.step()
		policy_optimizer.step()

		return value_loss.data.numpy()[0], policy_loss.data.numpy()[0]

	# calculate the advantages...
	def calculate_the_loss(self, reward_batch_tensor, predicted_value, terminal_batch, log_p_batch, previous_return):
		value_loss = 0
		policy_loss = 0

		for idx in reversed(range(len(terminal_batch))):
			if terminal_batch[idx]:
				returns = reward_batch_tensor[idx]
				advantages = returns - predicted_value[idx, 0]
				value_loss += (advantages ** 2) * 0.5

				# remove the advantage from the graph...
				advantages = advantages.detach()
				policy_loss += -log_p_batch[idx] * advantages

			else:
				returns = reward_batch_tensor[idx] + self.gamma * previous_return
				advantages = returns - predicted_value[idx, 0]
				value_loss += (advantages ** 2) * 0.5
				
				# remove the advantage from the graph...
				advantages = advantages.detach()
				policy_loss += -log_p_batch[idx] * advantages

			previous_return = returns
		
		return value_loss, policy_loss

	#  this is from https://github.com/ikostrikov/pytorch-a3c/
	def transfer_grads_to_shared_models(self, value_model, value_model_share, policy_model, policy_model_share):
		for param, shared_param in zip(value_model.parameters(), value_model_share.parameters()):
			if shared_param.grad is not None:
				return
			shared_param._grad = param.grad

		for param, shared_param in zip(policy_model.parameters(), policy_model_share.parameters()):
			if shared_param.grad is not None:
				return
			shared_param._grad = param.grad

	# -------------------------------- Test the network... ------------------------------------#
	def test_the_network(self, path):
		# load the models....
		self.policy_network_local.load_state_dict(torch.load(path))
		self.policy_network_local.eval()
		state = self.env.reset()
		reward_sum = 0
		while True:
			while True:
				self.env.render()
				state_tensor = torch.from_numpy(state).view(1, -1)
				action, _ = self.policy_network_local(Variable(state_tensor))
				action = action.data.numpy()[0]
				action = -2 + action * 4
				state_, reward, terminal, _ = self.env.step(action)
				reward_sum += reward

				if terminal:
					break
				state = state_

			state = self.env.reset()
			print('The reward_sum is ' + str(reward_sum))
			reward_sum = 0
