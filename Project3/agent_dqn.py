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
BATCH_SIZE = 32 # should be 32
REPLAY_BUFFER_SIZE = int(1e5) # should be 1e5
REWARD_BUFFER_SIZE = 30 # units of episodes
TARGET_Q_NET_UPDATE_PERIOD = int(5e3)
GAMMA = 0.99
LEARN_RATE = 5e-4
EPSILON_INITIAL = 1.0
EPSILON_FINAL = 1e-3
EPSILON_DECAY_DURATION = int(1e4)
LOG_PERIOD = int(1e4)
SAVE_INTERVAL = int(1e3) # try 1e4?
MODEL_SAVE_PATH_0 = './dqn_vanilla_0.pth'
MODEL_SAVE_PATH_1 = './dqn_vanilla_1.pth'
ANALYTICS_SAVE_PATH = './analytics.csv'
MODEL_LOAD_PATH = './dqn_vanilla.pth'
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
        - store the experience in the replay memory (should be store_experience())
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
4. add logs to check if:
    - gpu is used
    - how many episodes...?
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

        # Define variables for input arguments
        self.logging_enabled = args.logging_enabled or False
        self.save_path_alternate = 0 # will alternate between 0 and 1
        self.save_path = [MODEL_SAVE_PATH_0, MODEL_SAVE_PATH_1]
        self.load_path = MODEL_LOAD_PATH
        self.analytics_filename = ANALYTICS_SAVE_PATH

        # Prioritize GPU training
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Write device type
        with open ("DEVICE", "w") as device_file:
            device_str = str(self.device)
            device_file.write(device_str)
        
        # Check to test or load
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            loaded_state_dicts = self.load_model()
            self.initialize_network(loaded_state_dicts)

        elif args.resume_training:
            loaded_state_dicts = self.load_model()
            self.initialize_network(loaded_state_dicts)

        else: # start from scratch
            self.initialize_network()
                    
        # Initialize buffers
        self.replay_buffer_deque = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.reward_buffer_deque = deque([0.0], maxlen=30)
        self.avg_30_reward_lst = []

        self.initialize_replay_buffer() # fill up replay buffer
            
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
    
    def initialize_network(self, loaded_state_dict=None, test=False):
        """Initialize Q-value training/prediction network.

        Args:
            loaded_state_dict: Tensor object loaded by the torch.load function.
            test: Boolean to indicate whether the agent is in testing mode or not.
        """

        # Define neural networks
        self.training_nn = DQN(GAMMA).to(self.device) # initialize action-value function Q with random weights theta
        self.target_nn = DQN(GAMMA).to(self.device)

        # Define loss and optimizer
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.SmoothL1Loss()

        # self.optimizer = optim.SGD(self.training_nn.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)
        self.optimizer = optim.Adam(self.training_nn.parameters(), lr=LEARN_RATE)

        # Load the model if present
        if loaded_state_dict:
            self.training_nn.load_state_dict(loaded_state_dict['model_state_dict'])
            self.optimizer.load_state_dict(loaded_state_dict['optimizer_state_dict'])
            self.criterion = loaded_state_dict['loss']

        # Set the target network to the same initial parameters as the training/online network
        self.target_nn.load_state_dict(self.training_nn.state_dict())

        if test: self.training_nn.eval()

    def initialize_replay_buffer(self):
        """Fill up the replay buffer.
        """

        if self.logging_enabled: print('Initializing replay buffer...', end='')

        # Start the environment
        observation = self.env.reset()

        # Make the agent play the game
        for _ in range(REPLAY_BUFFER_SIZE):

            # Pick and execute action
            action = self.env.action_space.sample()
            next_observation, reward, done, _ = self.env.step(action)
            exp_tup = (observation, action, reward, done, next_observation)

            # Append 5-tuple to replay buffer
            self.store_experience(exp_tup)

            if done: observation = self.env.reset()
            else: observation = next_observation

        if self.logging_enabled: print('Done!\n')

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

        # Predict Q-values from observation
        q_val = self.training_nn(self.obs_arr_to_tensor(observation)) # create bogus batch dimension

        # Select highest scoring action
        max_q_ind = torch.argmax(q_val, dim=1)[0]
        action = max_q_ind.detach().item()
        
        ###########################
        return action
    
    def store_experience(self, experience_tuple):
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
        self.reward_buffer_deque.append(experience_tuple[2])
        
        ###########################        
        
    def sample_stored_experiences(self, batch_size):
        """ You can add additional arguments as you need.
        Select batch from buffer.

        Returns:
            A randomly sampled batch of experiences (in 5-tuple form)
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        return random.sample(self.replay_buffer_deque, batch_size)
        
        ###########################

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        # Main training loop
        observation = self.env.reset()
        episode_reward = 0.0
        episode_counter = 1

        # Run the episode for as long as we can
        while True:

            # Adjust epsilon to modify tendency for exploration vs exploitation
            epsilon = np.interp(episode_counter-1, [0, EPSILON_DECAY_DURATION], [EPSILON_INITIAL, EPSILON_FINAL])

            for _ in range(5): # only 5 lives

                # Select epsilon-greedy action
                if np.random.rand() <= epsilon:
                    action = self.env.action_space.sample()
                else:
                    # Use the trained neural net to pick an action with the max
                    # Q-value (i.e., using Q to guide policy)
                    action = self.training_nn( self.obs_arr_to_tensor(observation) ).argmax().item()

                # Execute selected action then store observation and reward in replay buffer
                next_observation, reward, done, _ = self.env.step(action)
                exp_tup = (observation, action, reward, done, next_observation)
                self.store_experience(exp_tup)

                # Accumulate total reward for current episode
                episode_reward += reward

                if done:
                    observation = self.env.reset()
                    self.reward_buffer_deque.append(episode_reward)
                    episode_reward = 0.0
                else: observation = next_observation

                """
                Experience sampling
                """
                # Gradient stepd
                # Randomly sample batches of experiences
                rnd_exp_tup_lst = self.sample_stored_experiences(BATCH_SIZE)

                # @todo: need to figure out the data format here
                # is the shape (batch_size, c, h, w)???

                # Parse sampled experiences
                rnd_observations = np.asarray([tup[0] for tup in rnd_exp_tup_lst])
                rnd_actions = np.asarray([tup[1] for tup in rnd_exp_tup_lst])
                rnd_rewards = np.asarray([tup[2] for tup in rnd_exp_tup_lst])
                rnd_dones = np.asarray([tup[3] for tup in rnd_exp_tup_lst])
                rnd_next_observations = np.asarray([tup[4] for tup in rnd_exp_tup_lst])

                rnd_observations_tensor = self.obs_arr_to_tensor(rnd_observations)
                rnd_actions_tensor = torch.as_tensor(rnd_actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
                rnd_rewards_tensor = torch.as_tensor(rnd_rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
                rnd_dones_tensor = torch.as_tensor(rnd_dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
                rnd_next_observations_tensor = self.obs_arr_to_tensor(rnd_next_observations)

                tensor_lst = [rnd_observations_tensor,
                            rnd_actions_tensor,
                            rnd_rewards_tensor,
                            rnd_dones_tensor,
                            rnd_next_observations_tensor]
                            
                # Update training network
                self.optimizer.zero_grad() # set gradients to zero                
                loss = self.training_nn.compute_loss(tensor_lst, self.target_nn, self.criterion) # compute loss                
                loss.backward() # backpropogate weights                
                torch.nn.utils.clip_grad_value_(self.training_nn.parameters(), 1) # clip gradients                
                self.optimizer.step()

            # Update target network parameters to 'catch up'
            if episode_counter % TARGET_Q_NET_UPDATE_PERIOD == 0:
                self.target_nn.load_state_dict(self.training_nn.state_dict())

            # Compute the 30-episode averaged reward
            if episode_counter % 30 == 0:
                self.most_recent_avg_30_reward = np.mean(self.reward_buffer_deque)
                self.avg_30_reward_lst.append(self.most_recent_avg_30_reward)
                
            # Log training performance
            if episode_counter % LOG_PERIOD == 0 and self.logging_enabled:
                print("Episode:", episode_counter)
                print("Last 30-episode averaged reward:", self.most_recent_avg_30_reward) # @todo: is this right? it only computes the mean for the length of available rewards...
                # print("Average reward:", np.mean(self.reward_buffer_arr))

            # Save data every interval
            if episode_counter % SAVE_INTERVAL == 0:
                self.save_model()
                self.save_analytics()

            episode_counter += 1 # update counter
        
        ###########################

    def obs_arr_to_tensor(self, obs):
        """Converts observations into PyTorch tensors with values between 0 to 1.0.

        Args:
            obs: A single or batch of 84x84x4 numpy array(s) for the Atari Breakout game

        Returns:
            A PyTorch tensor with dimensions (batch_size, channels, height, width) and values between 0 to 1.0.
        """

        assert(len(obs.shape) <= 4) # only allow single or batch

        # Check if it's just a single frame or minibatch
        if len(obs.shape) == 3:
            # obs = torch.as_tensor(obs, device=self.device).unsqueeze(0) # change to numpy array!
            obs = np.expand_dims(obs, axis=0)

        obs = np.array(obs).transpose(0, 3, 1, 2) # flip dimensions
        obs = torch.tensor(obs/255.0, dtype=torch.float32, device=self.device) # force values to be between 0 and 1, and of float32 type (default type)

        return obs

    def save_model(self):
        """Save Q-value neural network model.
        """
        
        # Alternate save path to prevent corrupted saves
        self.save_path_alternate = 1 - self.save_path_alternate

        torch.save({
            'model_state_dict': self.training_nn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.criterion,
        }, self.save_path[self.save_path_alternate])

    def load_model(self):
        """Load trained Q-value neural network model
        """

        return torch.load(self.load_path, map_location=self.device)

    def save_analytics(self):
        
        # Write to file then clear buffer
        with open (self.analytics_filename, "a") as a_file:
            str_lst = [str(i) + "\n" for i in self.avg_30_reward_lst]
            a_file.writelines(str_lst)
            self.avg_30_reward_lst = []