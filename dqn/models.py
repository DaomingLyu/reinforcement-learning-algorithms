import torch
import torch.nn as nn
import torch.nn.functional as F

# deep Q learning network...
class Deep_Q_Network(nn.Module):
    def __init__(self, num_actions):
        super(Deep_Q_Network, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.linear1 = nn.Linear(64 * 10 * 10, 512)
        self.action_value = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, 64 * 10 * 10)
        x = F.relu(self.linear1(x))
        q_value = self.action_value(x)

        max_q_value, action = torch.max(q_value, 1)

        max_q_value = max_q_value.detach()
        action = action.detach()

        return q_value, max_q_value, action



