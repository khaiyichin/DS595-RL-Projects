#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

# Default values (inspired by the Nature paper)
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 1e5
REWARD_BUFFER_SIZE = 30 # units of episodes
MIN_BUFFER_SIZE = 1e3
TARGET_Q_NET_UPDATE_PERIOD = 5e3
GAMMA = 0.99
ACT_REPEAT = 4
UPDATE_FREQ = 4
LEARN_RATE = 25e-4
MOMENTUM = 0.95
SQ_MOMENTUM = 0.95
MIN_SQ_GRAD = 1e-2
MIN_REPLAY_SIZE = 1e3
EPSILON_INITIAL = 1.0
EPSILON_FINAL = 1e-3
EPSILON_DECAY = 1e4
LOG_PERIOD = 1e4
"""
Steps:

1. initialize replay memory capacity
2. initialize neural network with random weights
3. for each episode in desired episodes:
    a. initialize a starting state in the environment
    b. for each time step in the game:
        - select an action (via exploration or exploitation)
        - execute action in the environment
        - observe the reward and the next state the agent ends up in
        - store the experience in the replay memory (should be push())
        - sample the random batch from the replay memory (should be replay_buffer())
        - preprocess the states from the batch
        - pass the batch of preprocessed states to the policy (neural) network (the DQN class)
        - calculate the loss between the output Q values and the target Q values.
            + requires a second pass to the network for the next state
        - gradient descent updates weights in the neural network (done internally using)
"""

"""
@todos:
1. deal with args: where do we get them from, what should it contain?
2. load/save models.
3. parallel training.
"""

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #

        self.env = env

        # Prioritize GPU training
        device = torch.device("cudo:0" if torch.cuda.is_available() else "cpu")

        # Define neural networks
        self.training_nn = DQN(self.env).to(device) # initialize action-value function Q with random weights theta
        self.target_nn = DQN(self.env).to(device)

        # Set the target network to the same initial parameters as the training/online network
        self.target_nn.load_state_dict(self.training_nn.state_dict())

        # Define loss and optimizer
        # self.loss_calculator = torch.nn.CrossEntropyLoss()
        self.loss_calculator = torch.nn.SmoothL1Loss()

        # self.optimizer = optim.SGD(self.training_nn.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)
        self.optimizer = optim.Adam(self.training_nn.parameters(), lr=LEARN_RATE)
        
        # Initialize buffers
        self.replay_buffer_deque = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.reward_buffer_deque = deque([0.0], max_len=5e2)
        self.reward_buffer_arr = np.array([0.0])

        self.replay_buffer(None, True) # fill up replay buffer

        # Define variables for input arguments
        self.logging_enabled = args.logging_enabled or False
        self.num_episodes_desired = args.num_episodes_desired
        self.timesteps_desired = args.timesteps_desired
        
        
        
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        # Convert 4-tuple to PyTorch tensor
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32)

        # Predict Q-values from observation
        q_val = self.training_nn(obs_tensor.unsqueeze[0]) # create bogus batch dimension

        # Select highest scoring action
        max_q_ind = torch.argmax(q_val, dim=1)[0]
        action = max_q_ind.detach().item()
        
        ###########################
        return action
    
    def push(self, experience_tuple):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list

        Args:
            experience_tuple: 5-tuple containing current state, current action, current reward, terminal state boolean, and next state.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        self.replay_buffer_deque.append(experience_tuple)
        # self.reward_buffer_deque.append(experience_tuple[2])
        self.reward_buffer_arr = np.append(self.reward_buffer_arr, experience_tuple[2])
        
        ###########################
        
        
    def replay_buffer(self, batch_size, init=False):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        # Initialize the replay buffer with experiences
        if init:

            # Start the environment
            observation = self.env.reset()

            # Make the agent play the game
            for _ in range(MIN_REPLAY_SIZE):

                # Pick and execute action
                action = self.env.action_space.sample()
                next_observation, reward, done, _ = self.env.step(action)
                exp_tup = (observation, action, reward, done, next_observation)

                # Append 4-tuple to replay buffer
                self.push(exp_tup)

                if done: observation = self.env.reset()
                else: observation = next_observation
        
            return # nothing needed for initialization
        else:
            return random.sample(self.replay_buffer_deque, batch_size)
            pass
        
        ###########################
        return 
        

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        # Main training loop
        observation = self.env.reset()
        episode_reward = 0.0

        for step in range(self.num_episodes_desired):

            # Adjust epsilon to modify tendency for exploration vs exploitation
            epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_INITIAL, EPSILON_FINAL])

            # Select epsilon-greedy action
            draw = np.random.rand()

            if draw <= epsilon:
                action = self.env.action_space.sample()
            else:
                # Use the trained neural net to pick an action with the max
                # Q-value (i.e., using Q to guide policy)
                action = max(self.training_nn(observation))

            # Execute selected action then store observation and reward in replay buffer
            next_observation, reward, done, _ = self.env.step(action)
            exp_tup = (observation, action, reward, done, next_observation)
            self.push(exp_tup)

            # Accumulate total reward for current episode
            episode_reward += reward

            if done:
                observation = self.env.reset()
                self.reward_buffer.append(episode_reward)
                episode_reward = 0.0
            else: observation = next_observation




            """
            Experience sampling
            """
            # Gradient step
            # Randomly sample batches of experiences
            rnd_exp_tup_lst = self.replay_buffer(BATCH_SIZE)

            rnd_observations = np.asarray([tup[0] for tup in rnd_exp_tup_lst])
            rnd_actions = np.asarray([tup[1] for tup in rnd_exp_tup_lst])
            rnd_rewards = np.asarray([tup[2] for tup in rnd_exp_tup_lst])
            rnd_dones = np.asarray(tup[3] for tup in rnd_exp_tup_lst)
            rnd_next_observations = np.asarray([tup[4] for tup in rnd_exp_tup_lst])

            # Convert numpy arrays to PyTorch tensors
            rnd_observations_tensor = torch.as_tensor(rnd_observations, dtype=torch.float32)
            rnd_actions_tensor = torch.as_tensor(rnd_actions, dtype=torch.int64).unsqueeze(-1)
            rnd_rewards_tensor = torch.as_tensor(rnd_rewards, dtype=torch.float32).unsqueeze(-1)
            rnd_dones_tensor = torch.as_tensor(rnd_dones, dtype=torch.float32).unsqueeze(-1)
            rnd_next_observations_tensor = torch.as_tensor(rnd_next_observations, dtype=torch.float32)



            """
            Target computation
            """
            # Compute targets
            target_q_vals = self.target_nn(rnd_next_observations_tensor)
            max_target_q_vals = target_q_vals.max(dim=1, keepdim=True)[0]
            
            target_q_vals = rnd_rewards_tensor + GAMMA*(1-rnd_dones_tensor)*max_target_q_vals # condensed piecewise function from the Nature paper

            # Compute Q-values based on actual actions
            q_vals = self.training_nn(rnd_observations_tensor)
            actual_q_vals = torch.gather(input=q_vals, dim=1, index=rnd_actions_tensor) # based on actual actions took

            # Compute loss between the new Q-value and the updated Q-value
            # loss = torch.nn.functional.smooth_l1_loss(actual_q_vals, target_q_vals)
            loss = self.loss_calculator(actual_q_vals, target_q_vals)
            
            """
            Network update
            """
            # Update training network
            self.optimizer.zero_grad() # set gradients to zero
            loss.backward() # backpropogate weights
            self.optimizer.step()

            # Update target network parameters to 'catch up'
            if step % TARGET_Q_NET_UPDATE_PERIOD == 0:
                self.target_nn.load_state_dict(self.training_nn.state_dict())
                
            # Log output
            if step % LOG_PERIOD and self.logging_enabled:
                print("Step:", step)
                print("Average reward:", np.mean(self.reward_buffer_deque)) # @todo: is this right? it only computes the mean for the length of available rewards...
                print("Average reward:", np.mean(self.reward_buffer_arr))

            # Update weights using gradient descent to minimize loss


        # Save model every interval

        # Log training performance
        
        
        ###########################

    def observation_to_tensor(self, obs):
        """Converts observations into PyTorch tensors.

        Args:
            obs: An 84x84x4 image observation for the Atari Breakout game.

        Returns:
            A PyTorch tensor.
        """
        pass