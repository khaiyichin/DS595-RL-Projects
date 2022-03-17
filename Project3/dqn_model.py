#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #

        # @todos: allow parameter changes..?

        self.in_channels = in_channels
        self.num_actions = num_actions

        # Define neural network structure according to the Nature paper
        self.pool = nn.MaxPool2d(2, 2) # shrink the 2d image by a factor of 0.5
        self.conv_1 = nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv_3 = nn.Conv2d(64, 64, kernel=3, stride=1)
        self.fc_1 = nn.Linear(64 * 7 * 7, 512)
        self.output_layer = nn.Linear(512, self.num_actions)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        """
        """

        # Go through literature to find good DQN structure

        x = F.relu(self.conv_1(x)) # 84x84x4 -> 20x20x32
        x = F.relu(self.conv_2(x)) # 20x20x32 -> 9x9x64
        x = F.relu(self.conv_3(x)) # 9x9x64 -> 7x7x64
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc_1(x))
        x = self.output_layer(x)

        ###########################
        return x

    def update_weights(self):
        pass

# maintain 2 networks: 1 training network (get Q values) and 1 target network (fix Q temporarily)