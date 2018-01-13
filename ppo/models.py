import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 400)
        self.affine2 = nn.Linear(400, 300)

        # alpha of the distribution...
        self.action_alpha = nn.Linear(300, num_outputs)
        self.action_alpha.weight.data.mul_(0.1)
        self.action_alpha.bias.data.mul_(0.0)

        self.action_beta = nn.Linear(300, num_outputs)
        self.action_beta.weight.data.mul_(0.1)
        self.action_beta.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))

        # make sure the alpha and beta is larger than 1... for unimodal beta distribution
        action_alpha = F.softplus(self.action_alpha(x)) + 1
        action_beta = F.softplus(self.action_beta(x)) + 1

        return action_alpha, action_beta

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
