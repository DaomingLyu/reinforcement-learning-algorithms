import torch
#import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist


# in this case, we use the beta distribution....
class Policy(nn.Module):
	def __init__(self, num_inputs, num_outputs):
		super(Policy, self).__init__()
		self.affine1 = nn.Linear(num_inputs, 64)
		self.affine2 = nn.Linear(64, 64)

		self.action_alpha = nn.Linear(64, num_outputs)
		self.action_alpha.weight.data.mul_(0.1)
		self.action_alpha.bias.data.mul_(0)

		self.action_beta = nn.Linear(64, num_outputs)
		self.action_beta.weight.data.mul_(0.1)
		self.action_beta.bias.data.mul_(0)

	def forward(self, x):
		x = F.tanh(self.affine1(x))
		x = F.tanh(self.affine2(x))

		action_alpha = F.softplus(self.action_alpha(x)) + 1
		action_beta = F.softplus(self.action_beta(x)) + 1

		action = dist.beta(action_alpha, action_beta)
		# remove it from the graph...
		action = action.detach()
		# calculate the log probability....
		log_p = dist.beta.log_pdf(action, action_alpha, action_beta)
		
		return action, log_p


class Value(nn.Module):
	def __init__(self, num_inputs):
		super(Value, self).__init__()
		self.affine1 = nn.Linear(num_inputs, 64)
		self.affine2 = nn.Linear(64, 64)

		self.value_head = nn.Linear(64, 1)
		self.value_head.weight.data.mul_(0.1)
		self.value_head.bias.data.mul_(0)

	def forward(self, x):
		x = F.tanh(self.affine1(x))
		x = F.tanh(self.affine2(x))

		state_value = self.value_head(x)

		return state_value


# ------------------------------------------------------------------------------------- #
# For Discrete Control Problem.... (Add the lstm....)
class Actor_Critic(nn.Module):
	def __init__(self, num_outputs, deterministic=False):
		super(Actor_Critic, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 8, stride=4, padding=1)
		self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

		self.lstm = nn.LSTMCell(64 * 9 * 9, 256)

		self.linear_action = nn.Linear(256, num_outputs)
		self.linear_action.weight.data.mul_(0.1)
		self.linear_action.bias.data.mul_(0)
		
		self.linear_value = nn.Linear(256, 1)
		self.linear_value.weight.data.mul_(0.1)
		self.linear_value.bias.data.mul_(0)

		self.lstm.bias_ih.data.fill_(0)
		self.lstm.bias_hh.data.fill_(0)

		self.deterministic = deterministic


	def forward(self, inputs):
		inputs, (hx, cx) = inputs 
		x = F.relu(self.conv1(inputs))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))

		x = x.view(-1, 64*9*9)

		hx, cx = self.lstm(x, (hx, cx))
		x = hx

		# get the state_value
		value = self.linear_value(x)

		# get the action value...
		action_prob = F.softmax(self.linear_action(x))

		# sample the action...
		cat_dist = dist.Categorical(action_prob, one_hot=False)

		if self.deterministic:
			_, action = torch.max(action_prob, 1)
			action = action.detach()
			action = action.unsqueeze(0)
		else:
			action = cat_dist()
		log_p = cat_dist.log_pdf(action)
				
		# calculate the entropy...
		entropy = -(action_prob * action_prob.log()).sum(1)
		return value, action, log_p, entropy, (hx, cx)
























