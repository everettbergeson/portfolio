# policy_iteration.py
"""Volume 2: Policy Function Iteration.
<Name>
<Class>
<Date>
"""
import scipy.linalg as la
import numpy as np
import gym
from gym import wrappers


# Intialize P for test example
#Left =0
#Down = 1
#Right = 2
#Up= 3

P = {s : {a: [] for a in range(4)} for s in range(4)}
P[0][0] = [(0, 0, 0, False)]
P[0][1] = [(1, 2, -1, False)]
P[0][2] = [(1, 1, 0, False)]
P[0][3] = [(0, 0, 0, False)]
P[1][0] = [(1, 0, -1, False)]
P[1][1] = [(1, 3, 1, True)]
P[1][2] = [(0, 0, 0, False)]
P[1][3] = [(0, 0, 0, False)]
P[2][0] = [(0, 0, 0, False)]
P[2][1] = [(0, 0, 0, False)]
P[2][2] = [(1, 3, 1, True)]
P[2][3] = [(1, 0, 0, False)]
P[3][0] = [(0, 0, 0, True)]
P[3][1] = [(0, 0, 0, True)]
P[3][2] = [(0, 0, 0, True)]
P[3][3] = [(0, 0, 0, True)]



# Problem 1
def value_iteration(P, nS ,nA, beta = 1, tol=1e-8, maxiter=3000):
    """Perform Value Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
       v (ndarray): The discrete values for the true value function.
       n (int): number of iterations
    """
    # Initialize
    done = False
    iters = 0
    v0 = 1

    while done == False:
        # Perform value iteration
        sa_vector = np.zeros(nA)
        for a in range(nA):
            for tuple_info in P[s][a]:
            # tuple_info is a tuple of (probability, next state, reward, done)
            p, s_, u, _ = tuple_info
            # sums up the possible end states and rewards with given action
            sa_vector[a] += (p * (u + beta * v0[s_]))
        #add the max value to the value function
        v1[s] = np.max(sa_vector)


        # Check if converged or reached maxiters
        norm = la.norm(v1 - v0)
        if norm < tol or iters > maxiter:
            done = True

        # Otherwise, iterate again!
        iters += 1
        v0 = v1

    return v, iters


# Problem 2
def extract_policy(P, nS, nA, v, beta = 1.0):
    """Returns the optimal policy vector for value function v

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        v (ndarray): The value function values.
        beta (float): The discount rate (between 0 and 1).

    Returns:
        policy (ndarray): which direction to move in from each square.
    """
    raise NotImplementedError("Problem 2 Incomplete")

# Problem 3
def compute_policy_v(P, nS, nA, policy, beta=1.0, tol=1e-8):
    """Computes the value function for a policy using policy evaluation.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        policy (ndarray): The policy to estimate the value function.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.

    Returns:
        v (ndarray): The discrete values for the true value function.
    """
    raise NotImplementedError("Problem 3 Incomplete")

# Problem 4
def policy_iteration(P, nS, nA, beta=1, tol=1e-8, maxiter=200):
    """Perform Policy Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
    	v (ndarray): The discrete values for the true value function
        policy (ndarray): which direction to move in each square.
        n (int): number of iterations
    """
    pi_k = 
    # Some stuff about pi
    for k in range(maxiter):
        vk1 = compute_policy_v(pi_k)
        pi_k1 = extract_policy(vk1)
        if la.norm(pi_k1 - pi_k) < tol:
            break
        pi_k = pi_k1
    
    return V_k1, pi_k1
    


# Problem 5 and 6
def frozen_lake(basic_case=True, M=1000, render=False):
    """ Finds the optimal policy to solve the FrozenLake problem

    Parameters:
    basic_case (boolean): True for 4x4 and False for 8x8 environemtns.
    M (int): The number of times to run the simulation using problem 6.
    render (boolean): Whether to draw the environment.

    Returns:
    vi_policy (ndarray): The optimal policy for value iteration.
    vi_total_rewards (float): The mean expected value for following the value iteration optimal policy.
    pi_value_func (ndarray): The maximum value function for the optimal policy from policy iteration.
    pi_policy (ndarray): The optimal policy for policy iteration.
    pi_total_rewards (float): The mean expected value for following the policy iteration optimal policy.
    """
    env_name = ""
    # if basic_case == True, do 4x4
    if basic_case==True:
        env_name = 'FrozenLake-v0'
    # Otherwise do 8x8
    else:
        env_name = 'FrozenLake8x8-v0'

    # Make environment
    env = gym.make(env_name).env
    # Find number of states and actions
    number_of_states = env.nS
    number_of_actions = env.nA
    # Get the dictionary with all the states and actions
    dictionary_P = env.P

    # Calculate the value function and policy for the environment
    # using value iteration and the policy and value function
    # generated by policy iteration


    env.close()


# Problem 6
def run_simulation(env, policy, render=True, beta = 1.0):
    """ Evaluates policy by using it to run a simulation and calculate the reward.

    Parameters:
    env (gym environment): The gym environment.
    policy (ndarray): The policy used to simulate.
    beta float: The discount factor.
    render (boolean): Whether to draw the environment.

    Returns:
    total reward (float): Value of the total reward received under policy.
    """
    # Put environment in starting state
    obs = env.reset()

    if render==True:
        env.render(mode = 'human')

    done = False
    obs = 0
    while done != True:
        # Take a step in the optimal direction and update variables
        obs, reward, done, _ = env.step(int(policy[obs]))