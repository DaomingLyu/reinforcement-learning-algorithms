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
	def __init__(self, env_name, action_space=[2, 3], test_mode=False, info=False, gamma=0.99, entropy_beta=0.01, global_update_step=20):
		self.env = gym.make(env_name)
		# init some other parameters....
		self.gamma = gamma
		self.entropy_beta = entropy_beta
		self.global_update_step = global_update_step
		self.info = info
		self.action_space = action_space
		self.test_mode = test_mode

		# define the network...
		self.actor_critic_local = models.Actor_Critic(len(self.action_space), self.test_mode)

	# train the network....
	def train_network(self, actor_critic_global, path):
		# define the optimizer...
		optimizer = optim.RMSprop(actor_critic_global.parameters(), lr=0.0001)
		self.actor_critic_local.train()
		t = 1
		eposide_num = 0
		reward_sum = 0
		state = self.env.reset()
		state = self.image_processing(state)

		cx = Variable(torch.zeros(1, 256))
		hx = Variable(torch.zeros(1, 256))

		while True:
			self.actor_critic_local.load_state_dict(actor_critic_global.state_dict())
			t_start = t
			terminal = False
			brain_memory = []
			if terminal:
				# set the initial cells and hidden layers.... for LSTM
				cx = Variable(torch.zeros(1, 256))
				hx = Variable(torch.zeros(1, 256))
			else:
				cx = Variable(cx.data)
				hx = Variable(hx.data)

			# here is the initial state.... need to process...
			while terminal == False and t - t_start < self.global_update_step:
				# convert it into tensor...
				state_tensor = torch.from_numpy(state).view(-1, 1, 80, 80)
				# conver it into Variable... put into graph...
				state_tensor = Variable(state_tensor)
				# input the state into the policy networl...
				predicted_value, action, log_p, entropy, (hx, cx) = self.actor_critic_local((state_tensor, (hx, cx)))
				# select the action...
				action_selected_index = action.data.numpy()[0, 0]
				state_, reward, terminal, _ = self.env.step(self.action_space[action_selected_index])

				# add the reward...
				reward_sum += reward
				t += 1
				# store the useful informaiton....
				brain_memory.append((predicted_value.squeeze(0), log_p, reward, terminal, entropy))

				# process the state....
				state_ = self.image_processing(state_)
				state = state_


			if terminal == False:
				state_tensor = torch.from_numpy(state).view(-1, 1, 80, 80)
				previous_return, _, _, _, _ = self.actor_critic_local((Variable(state_tensor), (hx, cx)))
				# remove it from the graph...
				previous_return = previous_return.detach()
				previous_return = previous_return.squeeze(0)
			else:
				previous_return = 0

			# start to udate the network....
			v_loss, p_loss = self.update_the_global_network(brain_memory, actor_critic_global, optimizer, previous_return)
			
			if terminal:
				# process the ....
				state = self.env.reset()
				state = self.image_processing(state)

				if self.info:
					if eposide_num % 1 == 0:
						print('The eposide_num is ' + str(eposide_num) + ', and the reward_sum is ' + str(reward_sum)
								 + ' and the value_loss is ' + str(v_loss) + ' and the policy_loss is ' + str(p_loss))
					if eposide_num % 10 == 0:
						save_path = path + 'policy_model_' + str(eposide_num) + '.pt'
						torch.save(actor_critic_global.state_dict(), save_path)
						print('----------------------------------------------------')
						print('model has been saved!')
						print('----------------------------------------------------')
				reward_sum = 0
				eposide_num += 1

	# this is used to update the global network...
	def update_the_global_network(self, brain_memory, actor_critic_global, optimizer, previous_return):
		predicted_value_batch = [element[0] for element in brain_memory]
		log_p_batch = [element[1] for element in brain_memory]

		reward_batch = np.array([element[2] for element in brain_memory])
		reward_batch_tensor = torch.from_numpy(reward_batch)
		reward_batch_tensor = Variable(reward_batch_tensor)

		terminal_batch = [element[3] for element in brain_memory]

		entropy_batch = [element[4] for element in brain_memory]

		# get the returns and the advantages....
		total_loss, value_loss, policy_loss = self.calculate_the_loss(reward_batch_tensor, predicted_value_batch, terminal_batch, previous_return, entropy_batch, log_p_batch)

		# start to update...
		optimizer.zero_grad()
		total_loss.backward()
		torch.nn.utils.clip_grad_norm(self.actor_critic_local.parameters(), 50)

		self.transfer_grads_to_shared_models(self.actor_critic_local, actor_critic_global)

		optimizer.step()

		return value_loss.data.numpy()[0], policy_loss.data.numpy()[0]

	# calculate the advantages...
	def calculate_the_loss(self, reward_batch_tensor, predicted_value_batch, terminal_batch, previous_return, entropy_batch, log_p_batch):
		value_loss = 0
		policy_loss = 0

		for idx in reversed(range(len(terminal_batch))):
			if terminal_batch[idx]:
				returns = reward_batch_tensor[idx]
				advantages = returns - predicted_value_batch[idx]
				value_loss += (advantages ** 2) * 0.5

				advantages = advantages.detach()
				policy_loss += -log_p_batch[idx] * advantages - entropy_batch[idx] * self.entropy_beta
			else:
				returns = reward_batch_tensor[idx] + self.gamma * previous_return
				advantages = returns - predicted_value_batch[idx]
				value_loss += (advantages ** 2) * 0.5

				advantages = advantages.detach()
				policy_loss += -log_p_batch[idx] * advantages - entropy_batch[idx] * self.entropy_beta

			previous_return = returns

		total_loss = value_loss + policy_loss
		return total_loss, value_loss, policy_loss

	#  this is from https://github.com/ikostrikov/pytorch-a3c/
	def transfer_grads_to_shared_models(self, value_model, value_model_share):
		for param, shared_param in zip(value_model.parameters(), value_model_share.parameters()):
			if shared_param.grad is not None:
				return
			shared_param._grad = param.grad


	def test_the_network(self, path):
		# load the models....
		self.actor_critic_local.load_state_dict(torch.load(path))
		self.actor_critic_local.eval()
		state = self.env.reset()
		state = self.image_processing(state)
		reward_sum = 0

		while True:
			cx = Variable(torch.zeros(1, 256))
			hx = Variable(torch.zeros(1, 256))	
			while True:
				self.env.render()
				state_tensor = torch.from_numpy(state).view(-1, 1, 80, 80)
				_, action, _, _, (hx, cx) = self.actor_critic_local((Variable(state_tensor), (hx, cx)))
				action_selected_index = action.data.numpy()[0, 0]
				state_, reward, terminal, _ = self.env.step(self.action_space[action_selected_index])
				reward_sum += reward

				if terminal:
					break
				# process the state....
				state_ = self.image_processing(state_)
				state = state_

			# process the state.....
			state = self.env.reset()
			state = self.image_processing(state)
			print('The reward_sum is ' + str(reward_sum))
			reward_sum = 0

	#http://karpathy.github.io/2016/05/31/rl/
	def image_processing(self, I):
		I = I[35:195]
		I = I[::2,::2,0] # downsample by factor of 2
		I[I == 144] = 0 # erase background (background type 1)
		I[I == 109] = 0 # erase background (background type 2)
		I[I != 0] = 1 # everything else (paddles, ball) just set to 1
		return I.astype(np.float)





