#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Temporal Difference
    In this problem, you will implememnt an AI player for cliffwalking.
    The main goal of this problem is to get familar with temporal diference algorithm.
    You could test the correctness of your code 
    by typing 'nosetests -v td_test.py' in the terminal.
    
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need. 
'''
#-------------------------------------------------------------------------

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
        You can use the function from project2-1
    """
    ############################
    # YOUR IMPLEMENTATION HERE #

    # Find action with highest Q score and separate from remaining actions
    max_action = np.argmax(Q[state]) # only choosing the first value if tied
    rem_action = [a for a in range(len(Q[state])) if a != max_action]

    # Compute probability for maximum action using epsilon greedy heuristic
    max_action_prob = 1 - epsilon + epsilon/nA

    # Draw action
    if np.random.uniform() > max_action_prob: action = np.random.choice(rem_action)
    else: action = max_action

    ############################
    return action

def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """On-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    Hints:
    -----
    You could consider decaying epsilon, i.e. epsilon = 0.99*epsilon during each episode.
    """
    
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    
    # Iterate for n_episodes
    for _ in range(n_episodes):

        # Define a decaying epsilon value
        epsilon = 0.99*epsilon

        # Initialize the environment
        observation = env.reset()
        done = False
        
        # Get an action from the epsilon_greedy policy
        action = epsilon_greedy(Q, observation, env.action_space.n, epsilon)

        # Iterate until termination condition reached (game ends)
        while not done:

            # Execute one step in the environment by taking an action
            next_observation, reward, done, _ = env.step(action)

            # Get another action from the epsilon_greedy policy
            next_action = epsilon_greedy(Q, next_observation, env.action_space.n, epsilon)

            # TD update
            # td_target
            target = reward + gamma*Q[next_observation][next_action]

            # td_error
            error = target - Q[observation][action]

            # Update Q
            Q[observation][action] = Q[observation][action] + alpha*error

            # Update state
            observation = next_observation

            # Update action
            action = next_action

    ############################
    return Q

def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """Off-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    """
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    
    # Iterate for n_episodes
    for _ in range(n_episodes):

        # Define a decaying epsilon value
        epsilon = 0.99*epsilon

        # Initialize the environment
        observation = env.reset()
        done = False
        
        # Iterate until termination condition reached (game ends)
        while not done:

            # Get an action from the epsilon_greedy policy
            action = epsilon_greedy(Q, observation, env.action_space.n, epsilon)
            
            # Execute one step in the environment by taking an action
            next_observation, reward, done, _ = env.step(action)
            
            # TD update
            # td_target with best Q
            target = reward + gamma*max(Q[next_observation])

            # td_error
            error = target - Q[observation][action]

            # Update Q
            Q[observation][action] = Q[observation][action] + alpha*error
            
            # update state
            observation = next_observation

    ############################
    return Q
