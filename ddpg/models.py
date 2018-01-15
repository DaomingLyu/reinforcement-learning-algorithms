import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class Policy(nn.Module):
	def __init__(self, num_input, num_action):
		super(Policy, self).__init__()
		self.affine1 = nn.Linear(num_input, 400)
		self.affine2 = nn.Linear(400, 300)
		self.action = nn.Linear(300, num_action)

		# init the first two layer... according to the origin paper....
		nn.init.uniform(self.affine1.weight, -np.sqrt(1 / num_input), np.sqrt(1 / num_input))
		nn.init.uniform(self.affine1.bias, -np.sqrt(1 / num_input), np.sqrt(1 / num_input))

		nn.init.uniform(self.affine2.weight, -np.sqrt(1 / 400), np.sqrt(1 / 400))
		nn.init.uniform(self.affine2.bias, -np.sqrt(1 / 400), np.sqrt(1 / 400))

		# init the last layer... according to the origin paper....
		nn.init.uniform(self.action.weight, -3e-3, 3e-3)
		nn.init.uniform(self.action.bias, -3e-3, 3e-3)

	def forward(self, x):
		x = F.relu(self.affine1(x))
		x = F.relu(self.affine2(x))

		action_out = F.tanh(self.action(x))

		return action_out

class Critic(nn.Module):
	def __init__(self, num_input, num_action):
		super(Critic, self).__init__()
		self.affine1 = nn.Linear(num_input, 400)
		self.affine2 = nn.Linear(400 + num_action, 300)
		self.value_head = nn.Linear(300, 1)

		# start to init....
		nn.init.uniform(self.affine1.weight, -np.sqrt(1 / num_input), np.sqrt(1 / num_input))
		nn.init.uniform(self.affine1.bias, -np.sqrt(1 / num_input), np.sqrt(1 / num_input))

		nn.init.uniform(self.affine2.weight, -np.sqrt(1 / (400 + num_action)), np.sqrt(1 / (400 + num_action)))
		nn.init.uniform(self.affine2.bias, -np.sqrt(1 / (400 + num_action)), np.sqrt(1 / (400 + num_action)))

		# init the last layer...
		nn.init.uniform(self.value_head.weight, -3e-3, 3e-3)
		nn.init.uniform(self.value_head.bias, -3e-3, 3e-3)

	def forward(self, x, action_out):
		x = F.relu(self.affine1(x))
		x = torch.cat((x, action_out), 1)
		x = F.relu(self.affine2(x))
		value = self.value_head(x)

		return value

