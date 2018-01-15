import numpy as np 
import torch
from torch.autograd import Variable
import models
from exploration_noise import OUNoise
import random
from running_state import ZFilter

# this is the implementation of thte DDPG(Deep Deterministic Policy Gradient)
# last update 2018-Jan-11
# author: Tianhong Dai
class ddpg_brain:
	def __init__(self, env, policy_lr, value_lr, tau, gamma, buffer_size, max_time_step, observate_time, batch_size, 
						path, soft_update_step, use_cuda):
		self.env = env
		self.policy_lr = policy_lr
		self.value_lr = value_lr
		self.use_cuda = bool(use_cuda)
		self.tau = tau
		self.gamma = gamma
		self.buffer_size = buffer_size
		self.max_time_step = max_time_step
		self.observate_time = observate_time
		self.batch_size = batch_size
		self.global_time_step = 0
		self.path = path
		self.soft_update_step = soft_update_step

		print('IF USE CUDA: ' + str(self.use_cuda))

		num_inputs = self.env.observation_space.shape[0]
		self.num_actions = self.env.action_space.shape[0]

		# the scale of the action space....
		self.action_scale = self.env.action_space.high[0]

		# build up the network....
		# build the actor_network firstly...
		self.actor_net = models.Policy(num_inputs, self.num_actions)
		self.actor_target_net = models.Policy(num_inputs, self.num_actions)

		# build the critic_network....
		self.critic_net = models.Critic(num_inputs, self.num_actions)
		self.critic_target_net = models.Critic(num_inputs, self.num_actions)

		# if use cuda...
		if self.use_cuda:
			self.actor_net.cuda()
			self.actor_target_net.cuda()

			self.critic_net.cuda()
			self.critic_target_net.cuda()

		# init the same parameters....
		self.actor_target_net.load_state_dict(self.actor_net.state_dict())
		self.critic_target_net.load_state_dict(self.critic_net.state_dict())

		# define the optimize.... add the L2 reg in critic optimzier here...
		self.optimizer_actor = torch.optim.Adam(self.actor_net.parameters(), lr=self.policy_lr)
		self.optimizer_critic = torch.optim.Adam(self.critic_net.parameters(), lr=self.value_lr, weight_decay=1e-2)

		# init the filter...
		self.running_state = ZFilter((num_inputs, ), clip=5)

	def train_network(self):
		# init the brain memory....
		brain_memory = []
		num_of_eposide = 0
		# init the noise for exploration...
		ou_noise = OUNoise(self.num_actions)
		reward_mean = None
		actor_loss = 0
		critic_loss = 0

		while True:
			reward_sum = 0
			# reset the noise....
			state = self.env.reset()
			state = self.running_state(state)
			ou_noise.reset()

			for t in range(self.max_time_step):
				state_tensor = torch.Tensor(state).view(1, -1)
				if self.use_cuda:
					state_tensor = Variable(state_tensor).cuda()
				else:
					state_tensor = Variable(state_tensor)

				# input the state to the actor network....
				action_out = self.actor_net(state_tensor)
				action_selected = self.select_actions(action_out, ou_noise)

				# input the action into the env...
				state_, reward, done, _ = self.env.step(self.action_scale * action_selected)

				# sum the reward...
				reward_sum += reward

				# use filter...
				state_ = self.running_state(state_)

				# store the transition....
				if len(brain_memory) >= self.buffer_size:
					brain_memory.pop(0)
				
				brain_memory.append((state, reward, done, state_, action_selected))

				# update the global_time_step...
				self.global_time_step += 1

				# the code style here not good...				
				if len(brain_memory) >= self.observate_time:
					mini_batch = random.sample(brain_memory, self.batch_size)
					critic_loss, actor_loss = self.update_networks(mini_batch)

				if done:
					break
				state = state_
			# we will use the expoential averaged reward: so it's the average of 1 / 0.01 = 100 games.
			reward_mean = reward_sum if reward_mean is None else reward_mean * 0.99 + reward_sum * 0.01

			if num_of_eposide % 10 == 0:
				print('The episode number is ' + str(num_of_eposide) + ', the running mean reward is ' + str(reward_mean) + 
					', the actor_loss is ' + str(actor_loss) + ', the critic_loss is ' + str(critic_loss))
			if num_of_eposide % 100 == 0:
				save_path = self.path + 'policy_model_' + str(num_of_eposide) + '.pt'
				torch.save([self.actor_net.state_dict(), self.running_state], save_path)

			num_of_eposide += 1

	# select the action....
	def select_actions(self, action_out, ou_noise):
		action_numpy = action_out.data.cpu().numpy()[0]
		action_selected = action_numpy + ou_noise.noise()
		action_selected = np.clip(action_selected, -1, 1)

		return action_selected


	# update the network....
	def update_networks(self, buffer_batch):
		state_batch = np.array([element[0] for element in buffer_batch])
		state_batch_tensor = torch.Tensor(state_batch)

		reward_batch = np.array([element[1] for element in buffer_batch])
		reward_batch_tensor = torch.Tensor(reward_batch).view(-1, 1)

		terminal_batch = np.array([int(element[2]) for element in buffer_batch])
		terminal_batch = 1 - terminal_batch
		terminal_batch_tensor = torch.Tensor(terminal_batch).view(-1, 1)

		state_next_batch = np.array([element[3] for element in buffer_batch])
		state_next_batch_tensor = torch.Tensor(state_next_batch)

		action_batch = np.array([element[4] for element in buffer_batch])
		action_batch_tensor = torch.Tensor(action_batch)

		# process the data....
		if self.use_cuda:
			state_batch_tensor = Variable(state_batch_tensor).cuda()
			reward_batch_tensor = Variable(reward_batch_tensor).cuda()
			terminal_batch_tensor = Variable(terminal_batch_tensor).cuda()
			state_next_batch_tensor = Variable(state_next_batch_tensor).cuda()
			action_batch_tensor = Variable(action_batch_tensor).cuda()
		else:
			state_batch_tensor = Variable(state_batch_tensor)
			reward_batch_tensor = Variable(reward_batch_tensor)
			terminal_batch_tensor = Variable(terminal_batch_tensor)
			state_next_batch_tensor = Variable(state_next_batch_tensor)
			action_batch_tensor = Variable(action_batch_tensor)

		# will up date the critic network... 
		critic_loss = self.update_critic_network(state_batch_tensor, state_next_batch_tensor, terminal_batch_tensor, 
												reward_batch_tensor, action_batch_tensor)

		# will update the actor network...
		actor_loss = self.update_actor_network(state_batch_tensor)

		# will soft update the target network....		
		self.soft_update_target_network(self.critic_target_net, self.critic_net)
		self.soft_update_target_network(self.actor_target_net, self.actor_net)

		return critic_loss.data.cpu().numpy()[0], actor_loss.data.cpu().numpy()[0]

	# update the critic network....
	def update_critic_network(self, state_batch_tensor, state_next_batch_tensor, terminal_batch_tensor, 
			reward_batch_tensor, action_batch_tensor):
		# calculate the target number...
		action_out = self.actor_target_net(state_next_batch_tensor)
		expected_Q = self.critic_target_net(state_next_batch_tensor, action_out)
		target = reward_batch_tensor + self.gamma * expected_Q * terminal_batch_tensor
		# detach from the calculation graphic....
		target = target.detach()
		# calculate the Q value...
		Q_value = self.critic_net(state_batch_tensor, action_batch_tensor)

		loss = (target - Q_value).pow(2).mean()

		self.optimizer_critic.zero_grad()
		
		loss.backward()
		self.optimizer_critic.step()

		return loss

	# update the actor network...
	def update_actor_network(self, state_batch_tensor):
		loss = -self.critic_net(state_batch_tensor, self.actor_net(state_batch_tensor))
		loss = loss.mean()

		self.optimizer_actor.zero_grad()
		loss.backward()
		self.optimizer_actor.step()

		return loss

	# soft update the target network....
	def soft_update_target_network(self, target, source):
		# update the critic network firstly...
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ------------------------------------------------------------------------------------------------------- #

	# here is used to test the network....
	def test_network(self, env_name):
		model_path = 'saved_models/' + env_name + '/policy_model.pt'
		actor_model, filter_model = torch.load(model_path, map_location=lambda storage, loc: storage)
		self.actor_net.load_state_dict(actor_model)
		self.actor_net.eval()
		while True:
			state = self.env.reset()
			state = self.test_filter(state, filter_model.rs.mean, filter_model.rs.std)
			reward_sum = 0

			while True:
				self.env.render()
				state_tensor = torch.Tensor(state).view(1, -1)
				state_tensor = Variable(state_tensor)
				action_out = self.actor_net(state_tensor)
				actor_numpy = action_out.data.numpy()[0]

				state_, reward, done, _ = self.env.step(self.action_scale * actor_numpy)
				state_ = self.test_filter(state_, filter_model.rs.mean, filter_model.rs.std)
				reward_sum += reward

				if done:
					break

				state = state_

			print('The reward sum is ' + str(reward_sum))

	# used for testing... reduce mean and the variance...
	def test_filter(self, x, mean, std, clip=10):
		x = x - mean
		x = x / (std + 1e-8)
		x = np.clip(x, -clip, clip)

		return x






















