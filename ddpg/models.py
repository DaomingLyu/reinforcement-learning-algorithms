import torch
import torch.nn as nn
import torch.nn.functional as F 

class Policy(nn.Module):
	def __init__(self, num_input, num_action):
		super(Policy, self).__init__()
		self.affine1 = nn.Linear(num_input, 400)
		self.affine2 = nn.Linear(400, 300)
		self.action = nn.Linear(300, num_action)

		# init...
		self.action.weight.data.mul_(0.1)
		self.action.bias.data.mul_(0.0)

	def forward(self, x):
		x = F.relu(self.affine1(x))
		x = F.relu(self.affine2(x))

		action_out = F.tanh(self.action(x))

		return action_out

class Critic(nn.Module):
	def __init__(self, num_input, num_action):
		super(Critic, self).__init__()
		self.affine1 = nn.Linear(num_input, 400)
		self.affine_action = nn.Linear(num_action, 400)
		self.affine2 = nn.Linear(400, 300)
		self.value_head = nn.Linear(300, 1)

		# init....
		self.value_head.weight.data.mul_(0.1)
		self.value_head.bias.data.mul_(0.0)

	def forward(self, x, action_out):
		x = F.relu(self.affine1(x))
		action = F.relu(self.affine_action(action_out))
		x = x + action
		x = F.relu(self.affine2(x))
		value = self.value_head(x)

		return value

