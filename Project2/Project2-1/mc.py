#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from logging import error
from os import stat
from turtle import st
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.
    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.

    You don't have to follow the comments to write your code. They are provided
    as hints in case you need.
'''
#-------------------------------------------------------------------------

def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and his otherwise

    Parameters:
    -----------
    observation

    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # get parameters from observation
    score, dealer_score, usable_ace = observation
    action = None
    if score >= 20:
        action = 0

    else: action = 1

    ############################
    return action

def mc_prediction(policy, env, n_episodes, gamma = 1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value

    Note: at the begining of each episode, you need initialize the environment using env.reset()
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)

    ############################
    # YOUR IMPLEMENTATION HERE #
    # Iterate for n_episodes
    for _ in range(n_episodes):

        # Initialize the environment and episode list
        observation = env.reset()
        done = False
        episode = []

        # Iterate until termination condition reached (game ends)
        while not done:

            # Select an action based on the given policy
            action = policy(observation)

            # Record the state and action
            episode.extend([observation, action])

            # Execute one step in the environment by taking an action
            observation, reward, done, _ = env.step(action)

            # Record the reward obtained from the action
            episode.append(reward)

        G = 0 # initial return

        # Obtain the total number of steps taken in the current episode
        num_steps = int(len(episode) / 3)

        # Iterate for each step in the recorded sample starting from the back
        for step in range(num_steps-1, -1, -1):

            # Compute the return from current step, G
            G = gamma * G + episode[3*step + 2] # the reward is the 3rd element in each step

            # Extract state for current step
            state = episode[3*step]

            # Check if state has occurred in earlier steps (or if it's the earliest step)
            if state not in episode[0:3*(step-1)] or step == 0:

                # Update returns
                returns_count[state] += 1
                returns_sum[state] += G 

                # Calculate average return for this state over all sampled episodes
                V[state] = returns_sum[state] / returns_count[state]

    ############################

    return V

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.

    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1

    Returns:
    --------
    action: int
        action based current state
    Hints:
    ------
    With probability (1 − epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #

    # Find action with highest Q score and separate from remaining actions
    # max_action = np.random.choice( np.flatnonzero( Q[state] == Q[state].max() ) ) # useful for breaking ties with equal probability
    max_action = np.argmax(Q[state]) # only choosing the first value if tied
    rem_action = [a for a in range(len(Q[state])) if a != max_action]

    # Compute probability for maximum action using epsilon greedy heuristic
    max_action_prob = 1 - epsilon + epsilon/nA

    # Draw action
    if np.random.uniform() > max_action_prob: action = np.random.choice(rem_action)
    else: action = max_action

    ############################
    return action

def mc_control_epsilon_greedy(env, n_episodes, gamma = 1.0, epsilon = 0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-(0.1/n_episodes) during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    ############################
    # YOUR IMPLEMENTATION HERE #

    # Iterate for n_episodes
    for _ in range(n_episodes):

        # define decaying epsilon
        epsilon = epsilon - (0.1/n_episodes)

        # Initialize the environment and episode list
        observation = env.reset()
        done = False
        episode = []

        # Iterate until termination condition reached (game ends)
        while not done:

            # Get an action from the epsilon_greedy policy
            action = epsilon_greedy(Q, observation, env.action_space.n, epsilon)

            # Record the state and action
            episode.extend([observation, action])

            # Execute one step in the environment by taking an action
            observation, reward, done, _ = env.step(action)

            # Record the reward obtained from the action
            episode.append(reward)

        G = 0 # initial return

        # Obtain the total number of steps taken in the current episode
        num_steps = int(len(episode) / 3)

        # Iterate for each step in the recorded sample starting from the back
        for step in range(num_steps-1, -1, -1):

            # Compute the return from current step, G
            G = gamma * G + episode[3*step + 2] # the reward is the 3rd element in each step

            # Extract state action pair
            state = episode[3*step]
            action = episode[3*step+1]
            state_action_pair = (state, action)

            # Define list of remaining state action pairs
            state_action_hist = [(episode[3*s], episode[3*s+1]) for s in range(step)]

            # Check if state action pair has occurred in earlier steps
            if state_action_pair not in state_action_hist:

                # Update returns
                returns_count[state_action_pair] += 1
                returns_sum[state_action_pair] += G

                # Calculate average return for this state action pair over all sampled episodes
                Q[state][action] = returns_sum[state_action_pair] / returns_count[state_action_pair]

    return Q
