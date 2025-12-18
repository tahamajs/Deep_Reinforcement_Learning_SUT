import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """Shared Actor-Critic network for A3C algorithm."""

    def __init__(self, input_channels, action_space):
        super(ActorCritic, self).__init__()
        self.feat_size = 64 * 7 * 7

        # Shared convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        # Policy head
        self.policy_fc = nn.Linear(self.feat_size, 256)
        self.policy_out = nn.Linear(256, action_space)

        # Value head
        self.value_fc = nn.Linear(self.feat_size, 256)
        self.value_out = nn.Linear(256, 1)

    def forward(self, x):
        # Shared feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.feat_size)

        # Policy output
        policy = F.relu(self.policy_fc(x))
        policy = F.log_softmax(self.policy_out(policy), dim=1)

        # Value output
        value = F.relu(self.value_fc(x))
        value = self.value_out(value)

        return policy, value
