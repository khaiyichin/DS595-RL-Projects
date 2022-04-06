#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

## Local imports ##
from datetime import datetime
###################

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

# Default values
DOUBLE_DQN = False
BATCH_SIZE = 32 # should be 32
REPLAY_BUFFER_SIZE = int(1e5) # units of episodes
REWARD_BUFFER_SIZE = 30 # units of episodes
TARGET_Q_UPDATE_PERIOD = int(5e3) # units of lives (i.e., 5 lives per episode, so units of episodes/5)
GAMMA = 0.99
LEARN_RATE = 5e-4
EPSILON_INITIAL = 1.0
EPSILON_FINAL = 1e-3
EPSILON_DECAY_DURATION = int(1e4) # units of episodes
LOG_PERIOD_INITIAL = int(1e4) # units of episodes
LOG_PERIOD_FINAL = int(1e4) # units of episodes
LOG_PERIOD_DIM_FACTOR = int(2) # diminishing factor, i.e., divisor
SAVE_INTERVAL_INITIAL = int(5e2) # units of episodes
SAVE_INTERVAL_FINAL = int(5e2) # units of episodes
SAVE_INTERVAL_DIM_FACTOR = int(2) # diminishing factor, i.e., divisor
INPUT_FOLDER = './'
OUTPUT_FOLDER = './'
MODEL_SAVE_NAME = 'dqn_vanilla_save.pth'
MODEL_LOAD_NAME = 'dqn_vanilla.pth'
ANALYTICS_SAVE_NAME = 'analytics.csv'
REWARD_LOG_SAVE_NAME = 'reward_log.csv'

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

        # Check to read input configuration file or use default values
        if args.config:
            self.read_yaml_config(args.config)

        else:
            # Initialize default config values
            self.double_dqn = DOUBLE_DQN
            self.batch_size = BATCH_SIZE
            self.replay_buffer_size = REPLAY_BUFFER_SIZE
            self.reward_buffer_size = REWARD_BUFFER_SIZE
            self.target_q_update_period = TARGET_Q_UPDATE_PERIOD
            self.gamma = GAMMA
            self.learn_rate = LEARN_RATE
            self.epsilon_i = EPSILON_INITIAL
            self.epsilon_f = EPSILON_FINAL
            self.epsilon_decay_duration = EPSILON_DECAY_DURATION
            self.log_period_initial = LOG_PERIOD_INITIAL
            self.log_period_final = LOG_PERIOD_FINAL
            self.log_period_dim_factor = LOG_PERIOD_DIM_FACTOR
            self.save_interval_initial = SAVE_INTERVAL_INITIAL
            self.save_interval_final = SAVE_INTERVAL_FINAL
            self.save_interval_dim_factor = SAVE_INTERVAL_DIM_FACTOR

            self.optimizer_type_str = 'adam' # choice between adam, rmsprop

            self.input_folder = INPUT_FOLDER
            self.output_folder = OUTPUT_FOLDER
            self.model_save_name = MODEL_SAVE_NAME
            self.model_load_name = MODEL_LOAD_NAME
            self.analytics_save_name = ANALYTICS_SAVE_NAME
            self.reward_log_save_name = REWARD_LOG_SAVE_NAME
            pass

        # Initialize periods
        self.log_period = self.log_period_initial
        self.save_interval = self.save_interval_initial

        # Create output folder if it doesn't exist already
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Prioritize GPU training
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Write device type
        with open (os.path.join(self.output_folder, "DEVICE"), "w") as device_file:
            device_str = str(self.device)
            device_file.write(device_str + '\n')
            if torch.cuda.is_available():
                device_file.write(str(torch.cuda.get_device_name(0)) + '\n')
        
        # Check to test or train
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            loaded_state_dicts = self.load_model(args.test_model_file_path)
            self.initialize_network(loaded_state_dicts, True)

        else: # train

            # Check to load or start from scratch
            if args.resume_training:

                if self.logging_enabled:
                    print("\nResuming training using", os.path.join(self.input_folder, self.model_load_name), "\n")
                    sys.stdout.flush()

                loaded_state_dicts = self.load_model()
                self.initialize_network(loaded_state_dicts)

            else: # start from scratch
                self.initialize_network()

            # Initialize buffers
            self.replay_buffer_deque = deque(maxlen=self.replay_buffer_size)
            self.reward_buffer_deque = deque(maxlen=self.reward_buffer_size)
            self.avg_30_reward_lst = []
            self.reward_log = []

            self.initialize_replay_buffer() # fill up replay buffer

    def read_yaml_config(self, config_file):

        import yaml
        yaml_config = None

        # Read YAML configuration file
        with open(config_file, 'r') as fopen:
            try:
                yaml_config = yaml.safe_load(fopen)
            except yaml.YAMLError as exception:
                print(exception)

        # Display input arguments
        print('\n\t' + '='*50 + ' Inputs ' + '='*50 + '\n')
        print('\tConfig file\t: ' + os.path.abspath(config_file))
        print('\n\t' + '='*48 + ' End Inputs ' + '='*48  + '\n')

        # Displayed processed arguments
        print('\n\t\t\t\t\t' + '='*15 + ' Processed Config ' + '='*15 + '\n')
        print('\t\t\t\t\t\t', end='')
        for line in yaml.dump(yaml_config, indent=4, default_flow_style=False):
            if line == '\n':
                print(line, '\t\t\t\t\t\t',  end='')
            else:
                print(line, end='')
        print('\r', end='') # reset the cursor for print
        print('\n\t\t\t\t\t' + '='*13 + ' End Processed Config ' + '='*13  + '\n')

        # Assign configuration values
        self.double_dqn = yaml_config["doubleDQN"]
        self.batch_size = int(yaml_config["batchSize"])
        self.replay_buffer_size = int(yaml_config["bufferSizes"]["replayBuffer"])
        self.reward_buffer_size = int(yaml_config["bufferSizes"]["rewardBuffer"])
        self.target_q_update_period = int(yaml_config["targetQUpdatePeriod"])
        self.gamma = yaml_config["gamma"]

        # Assign optimizer parameters
        self.optimizer_type_str = yaml_config["optimizer"]["type"].lower()
        self.learn_rate = yaml_config["optimizer"]["learnRate"]

        if self.optimizer_type_str == "rmsprop":
            self.gradient_momentum = yaml_config["optimizer"]["gradientMomentum"]
            self.sq_gradient_momentum = yaml_config["optimizer"]["sqGradientMomentum"]
            self.min_sq_gradient = yaml_config["optimizer"]["minSqGradient"]

        elif self.optimizer_type_str == "adam":
            # nothing special for now
            pass
            
        elif self.optimizer_type_str == "sgd":
            self.gradient_momentum = yaml_config["optimizer"]["gradientMomentum"]

        else:
            print("ERROR: UNKNOWN OPTIMIZER!")
            exit()

        # Epsilon parameters
        self.epsilon_i = yaml_config["exploration"]["epsilonInitial"]
        self.epsilon_f = yaml_config["exploration"]["epsilonFinal"]
        self.epsilon_decay_duration = int(yaml_config["exploration"]["epsilonDecayDuration"])

        # Logging and saving parameters
        self.log_period_initial = int(yaml_config["periods"]["logPeriodInitial"])
        self.log_period_final = int(yaml_config["periods"]["logPeriodFinal"])
        self.log_period_dim_factor = int(yaml_config["periods"]["logPeriodDimFactor"])
        self.save_interval_initial = int(yaml_config["periods"]["saveIntervalInitial"])
        self.save_interval_final = int(yaml_config["periods"]["saveIntervalFinal"])
        self.save_interval_dim_factor = int(yaml_config["periods"]["saveIntervalDimFactor"])

        if self.log_period_dim_factor < 1 or self.save_interval_dim_factor < 1:
            print("ERROR: Diminishing factor must be >= 1!")
            exit()

        if self.log_period_final > self.log_period_initial or self.save_interval_final > self.save_interval_initial:
            print("ERROR: Final period/interval values must be smaller than the initial values!")
            exit()

        # Path parameters
        self.input_folder = yaml_config["paths"]["inputFolder"]
        self.output_folder = yaml_config["paths"]["outputFolder"]
        self.model_save_name = yaml_config["paths"]["modelSaveName"]
        self.model_load_name = yaml_config["paths"]["modelLoadName"]
        self.analytics_save_name = yaml_config["paths"]["analyticsSaveName"]
        self.reward_log_save_name = yaml_config["paths"]["rewardLogSaveName"]

        sys.stdout.flush()

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
        self.training_nn = DQN(self.gamma, self.double_dqn).to(self.device) # initialize action-value function Q with random weights theta

        # Define loss and optimizer
        self.criterion = torch.nn.SmoothL1Loss()

        if self.optimizer_type_str == "rmsprop":
            self.optimizer = optim.RMSprop(self.training_nn.parameters(), lr=self.learn_rate, momentum=self.gradient_momentum, eps=self.min_sq_gradient)

        elif self.optimizer_type_str == "adam":
            self.optimizer = optim.Adam(self.training_nn.parameters(), lr=self.learn_rate)

        elif self.optimizer_type_str == "sgd":
            self.optimizer = optim.SGD(self.training_nn.parameters(), lr=self.learn_rate, momentum=self.gradient_momentum)

        else:
            print("ERROR: UKNOWN OPTIMIZER TYPE = " + self.optimizer_type_str)

        # Load the model if present
        if loaded_state_dict:

            if self.logging_enabled:
                print("Loading model... ", end="")
                sys.stdout.flush()

            self.training_nn.load_state_dict(loaded_state_dict['model_state_dict'])
            self.optimizer.load_state_dict(loaded_state_dict['optimizer_state_dict'])
            self.criterion = loaded_state_dict['loss']

            self.training_nn.train() # set the module to be in training mode

            if self.logging_enabled:
                print("Done!\n")
                sys.stdout.flush()

        # Check whether in testing or training mode
        if test:
            self.training_nn.eval() # set the module to be in evaluation mode

        else:
            # Set the target network to the same initial parameters as the training/online network
            self.target_nn = DQN(self.gamma, self.double_dqn).to(self.device)
            self.target_nn.load_state_dict(self.training_nn.state_dict())

    def initialize_replay_buffer(self):
        """Fill up the replay buffer.
        """

        if self.logging_enabled:
            print("Initializing replay buffer... ", end="")
            sys.stdout.flush()

        # Start the environment
        observation = self.env.reset()

        # Make the agent play the game
        for _ in range(self.replay_buffer_size):

            # Pick and execute action
            action = self.env.action_space.sample()
            next_observation, reward, done, _ = self.env.step(action)
            exp_tup = (observation, action, reward, done, next_observation)

            # Append 5-tuple to replay buffer
            self.store_experience(exp_tup)

            if done: observation = self.env.reset()
            else: observation = next_observation

        if self.logging_enabled:
            print("Done!\n")
            sys.stdout.flush()

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
        episode_counter = 1

        # Run the episode for as long as we can
        while True:

            # Adjust epsilon to modify tendency for exploration vs exploitation
            epsilon = np.interp(episode_counter-1, [0, self.epsilon_decay_duration], [self.epsilon_i, self.epsilon_f])

            episode_reward = 0.0

            # Play the game until all lives are used up
            lives_counter = 5 # only 5 lives

            while lives_counter > 0:

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
                    lives_counter -= 1
                else: observation = next_observation

                """
                Experience sampling
                """
                # Gradient stepd
                # Randomly sample batches of experiences
                rnd_exp_tup_lst = self.sample_stored_experiences(self.batch_size)

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
            
            self.reward_buffer_deque.append(episode_reward) # accumulate all rewards in one episode
            self.reward_log.append(episode_reward)

            # Update target network parameters to 'catch up' (consider each life as one 'cycle')
            if episode_counter*5 % self.target_q_update_period == 0:
                self.target_nn.load_state_dict(self.training_nn.state_dict())

            # Compute the 30-episode averaged reward
            if episode_counter % 30 == 0:
                self.most_recent_avg_30_reward = np.mean(self.reward_buffer_deque)
                self.avg_30_reward_lst.append(self.most_recent_avg_30_reward)
                
            # Log training performance
            if episode_counter % self.log_period == 0 and self.logging_enabled:
                print("\nEpisode:", episode_counter)
                print("Loss:", loss.item())
                try: print("Last 30-episode averaged reward:", self.most_recent_avg_30_reward)
                except: pass

                sys.stdout.flush()

                # Reduce frequency of log output
                if self.log_period > self.log_period_final:
                    self.log_period = int(self.log_period / self.log_period_dim_factor)

                    # Ensure that the final log period is the minimum specified value
                    if self.log_period < self.log_period_final: self.log_period = self.log_period_final

            # Save data every interval
            if episode_counter % self.save_interval == 0:
                self.save_model(episode_counter)
                self.save_analytics()

                # Reduce frequency of save interval
                if self.save_interval > self.save_interval_final:
                    self.save_interval = int(self.save_interval / self.save_interval_dim_factor)

                    # Ensure that the final save interval is the minimum specified value
                    if self.save_interval < self.save_interval_final: self.save_interval = self.save_interval_final

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

    def save_model(self, curr_episode_num):
        """Save Q-value neural network model.
        """

        # Add time and episode information to filename
        curr_time = datetime.now()

        split_path = os.path.splitext(os.path.join(self.output_folder, self.model_save_name))

        model_save_path = split_path[0] + "_eps" + str(curr_episode_num) + "_" + curr_time.strftime("%m%d%y_%H%M") + split_path[1]

        torch.save({
            'model_state_dict': self.training_nn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.criterion,
        }, model_save_path)

    def load_model(self, test_model_file=None):
        """Load trained Q-value neural network model
        """
        if test_model_file:
            return torch.load( test_model_file, map_location=self.device )
        else:
            return torch.load( os.path.join(self.input_folder, self.model_load_name), map_location=self.device )

    def save_analytics(self):
        
        # Write to file (if list is populated) then clear buffer
        if self.avg_30_reward_lst:
            with open (os.path.join(self.output_folder, self.analytics_save_name), "a") as a_file:
                str_lst = [str(i) + "\n" for i in self.avg_30_reward_lst]
                a_file.writelines(str_lst)
                self.avg_30_reward_lst = []

        with open(os.path.join(self.output_folder, self.reward_log_save_name), "a") as r_file:
            str_lst = [str(i) + "\n" for i in self.reward_log]
            r_file.writelines(str_lst)
            self.reward_log = []
