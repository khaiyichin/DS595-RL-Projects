### MDP Value Iteration and Policy Iteration
### Reference: https://web.stanford.edu/class/cs234/assignment1/index.html 
from multiprocessing.dummy import current_process
import numpy as np

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    
    value_function = np.zeros(nS)
    ############################
    # YOUR IMPLEMENTATION HERE #

    val_func_change = 1 # loop termination condition

    # Iterate until no significatn changes to the value function
    while val_func_change > tol:

        val_func_change = 0 # reset loop termination condition

        # Iterate through all states
        for state in range(nS):
            v_k = value_function[state] # current value function at iteration k (for current state)
            v_k_plus_1 = 0 # future value function at iteration k+1 (for current state)

            # Iterate through all the actions (for the current state)
            for action in range(nA):
                p_tuples = P[state][action] # p_tuples can contain many tuples if env is stochastic

                # Initialize 2nd half of the Bellman equation product (whereby term_1*term_2 = value function)
                # I.e., the sum of the product between the transition probability and the sum of the current reward and discounted future returns) in the bellman equation
                # The 1st half is simply the sum of the policy functions
                term_2 = 0

                # Iterate through all the next states
                for tup in p_tuples:
                    trans_prob, next_state, reward, _ = tup
                    term_2 += trans_prob * (reward + gamma * value_function[next_state]) # accumulate the later term (in the product); see equation for more info

                v_k_plus_1 += policy[state][action] * term_2

            value_function[state] = v_k_plus_1

            # Adjust termination condition
            val_func_change = max( abs(v_k_plus_1 - v_k), val_func_change)
    
    ############################
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray[nS]
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.
    """

    new_policy = np.zeros([nS, nA])
	############################
	# YOUR IMPLEMENTATION HERE #

    # Iterate through all the states
    for state in range(nS):
        q = np.zeros(nA) # initialize array of state-action values (q)

        # Iterate through all the actions based on current state
        for action in range(nA):
            p_tuples = P[state][action]

            # Iterate through all the next states based on current state
            for tup in p_tuples:
                trans_prob, next_state, reward, _ = tup

                q[action] += trans_prob * ( reward + gamma * value_from_policy[next_state] ) # accumulate the state-action value for current action and state

        # Assign new policy probability for action with highest state-action value
        new_policy[state][np.argmax(q)] = 1.0

	############################
    return new_policy


def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
    new_policy = policy.copy()
	############################
	# YOUR IMPLEMENTATION HERE #

    policy_stable = False
    current_value_func = np.zeros(nS)

    # Iterate until policy stops improving
    while not policy_stable:

        current_value_func = policy_evaluation(P, nS, nA, policy, gamma, tol) # compute value function for current policy
        new_policy = policy_improvement(P, nS, nA, current_value_func, gamma) # improve policy based on computed value function

        # Check if there are updates to new_policy
        if np.allclose(new_policy, policy):
            policy_stable = True
        else:
            policy = new_policy

    V = current_value_func # assign the last computed value function

	############################
    return new_policy, V

def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """
    
    V_new = V.copy()
    ############################
    # YOUR IMPLEMENTATION HERE #

    ############################
    return policy_new, V_new

def render_single(env, policy, render = False, n_episodes=100):
    """
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game. 
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    """
    total_rewards = 0
    for _ in range(n_episodes):
        ob = env.reset() # initialize the episode
        done = False
        while not done:
            if render:
                env.render() # render the game
            ############################
            # YOUR IMPLEMENTATION HERE #
            
    return total_rewards



