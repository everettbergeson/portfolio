# opengym.py
"""Volume 2: Open Gym
<Name>
<Class>
<Date>
"""

import gym
import numpy as np
from IPython.display import clear_output
import random

def find_qvalues(env,alpha=.1,gamma=.6,epsilon=.1):
    """
    Use the Q-learning algorithm to find qvalues.

    Parameters:
        env (str): environment name
        alpha (float): learning rate
        gamma (float): discount factor
        epsilon (float): maximum value

    Returns:
        q_table (ndarray nxm)
    """
    # Make environment
    env = gym.make(env)
    # Make Q-table
    q_table = np.zeros((env.observation_space.n,env.action_space.n))

    # Train
    for i in range(1,100001):
        # Reset state
        state = env.reset()

        epochs, penalties, reward, = 0,0,0
        done = False

        while not done:
            # Accept based on alpha
            if random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            # Take action
            next_state, reward, done, info = env.step(action)

            # Calculate new qvalue
            old_value = q_table[state,action]
            next_max = np.max(q_table[next_state])

            new_value = (1-alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            # Check if penalty is made
            if reward == -10:
                penalties += 1

            # Get next observation
            state = next_state
            epochs += 1

        # Print episode number
        if i % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")

    print("Training finished.")
    return q_table

# Problem 1
def random_blackjack(n):
    """
    Play a random game of Blackjack. Determine the
    percentage the player wins out of n times.

    Parameters:
        n (int): number of iterations

    Returns:
        percent (float): percentage that the player
                         wins
    """
    # Open environment
    wins = 0
    env = gym.make("Blackjack-v0")

    for i in range(n):
        # Take random step
        result = env.step(env.action_space.sample())

        # If the game didn't terminate, keep playing
        while result[2] != True:
            result = env.step(env.action_space.sample())
        
        # Count wins
        if result[1] == 1:
            wins += 1

        # Reset environment
        env.reset()

    # Return win percentage
    return wins/n
        

# Problem 2
def blackjack(n=11):
    """
    Play blackjack with naive algorithm.

    Parameters:
        n (int): maximum accepted player hand

    Return:
        percent (float): percentage of 10000 iterations
                         that the player wins
    """
    env = gym.make("Blackjack-v0")
    wins = 0
    
    for i in range(10000):
        # Get the inital hand
        obs = env.reset()
        done = False

        while not done:
            # If we haven't reached maximum, keep drawing
            if obs[0] <= n:
                obs, reward, done, info = env.step(1)
            # If we've reached the maximum, see what the dealer got
            else:
                obs, reward, done, info = env.step(0)

        # Add up win if we won
        if reward == 1:
            wins += 1
    env.close()

    # Return win percentage
    return wins/10000
    


        
# Problem 3
def cartpole():
    """
    Solve CartPole-v0 by checking the velocity
    of the tip of the pole

    Return:
        iterations (integer): number of steps or iterations
                              to solve the environment
    """
    env = gym.make("CartPole-v0")
    # Take random actions and visualize each action
    try:
        obs = env.reset()
        steps = 0
        done = False
        while not done:
            env.render()
            if obs[3] < 0:
                obs, reward, done, info = env.step(0)
            else:
                obs, reward, done, info = env.step(1)
            steps += 1
            if done:
                break
    finally:
        env.close()
    return steps
    

# Problem 4
def car():
    """
    Solve MountainCar-v0 by checking the position
    of the car.

    Return:
        iterations (integer): number of steps or iterations
                              to solve the environment
    """
    env = gym.make("MountainCar-v0")
    steps = 0
    # Take random actions and visualize each action
    try:
        obs = env.reset()
        done = False
        while not done:
            steps += 1
            env.render()
            # If we're moving backwards, move more backwards
            # until we're overpowered by gravity
            if obs[1] < 0:
                obs, reward, done, info = env.step(0)
            # If we're at 0, go into neutral to start rolling one way
            # or the other
            elif obs[1] == 0:
                obs, reward, done, info = env.step(1)
            # Otherwise go forwards
            else:
                obs, reward, done, info = env.step(2)
            if done:
                break
    finally:
        env.close()
    return steps
    

# Problem 5
def taxi(q_table):
    """
    Compare naive and q-learning algorithms.

    Parameters:
        q_table (ndarray nxm): table of qvalues

    Returns:
        naive (flaot): mean reward of naive algorithm
                       of 10000 runs
        q_reward (float): mean reward of Q-learning algorithm
                          of 10000 runs
    """
    env = gym.make("Taxi-v3")

    # Run with random actions
    total_reward = 0
    for i in range(10000):
        # Initialize
        current_reward = 0
        env.reset()
        done = False

        # Take random actions
        while not done:
            obs, reward, done, info = env.step(env.action_space.sample())
            current_reward += reward

        # Add up the rewards
        total_reward += current_reward
    # Average reward
    naive = total_reward/10000

    # Use q_table to determine optimal decision making
    q_reward = 0
    for i in range(10000):
        # Initialize
        curr_q_reward = 0
        obs = env.reset()
        done = False

        while not done:
            # Make decisions using q_table
            k = np.argmax(q_table[obs])
            obs, reward, done, info = env.step(k)
            curr_q_reward += reward
        
        # Add up rewards
        q_reward += curr_q_reward
    # Average reward
    q_reward = q_reward/10000

    return naive, q_reward