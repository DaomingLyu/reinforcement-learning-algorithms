import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Will use the Normal Distribution In this Case,

Because It's hard to calculate the KL-Divergence of the Beta Distribution

Date: 2018-04-04

"""

class Policy(nn.Module):
    def __init__(self, num_input, num_action):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_input, 400)
        self.affine2 = nn.Linear(400, 300)
        # get the alpha and beta from the beta distribution
        self.action_mean = nn.Linear(300, num_action)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_action))

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        
        # get the mean of the normal distribution
        action_mean = self.action_mean(x)
        # get the std of the normal distribution
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        return action_mean, action_std

# define the value network...
class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 400)
        self.affine2 = nn.Linear(400, 300)
        self.value_head = nn.Linear(300, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))

        state_values = self.value_head(x)
        return state_values
