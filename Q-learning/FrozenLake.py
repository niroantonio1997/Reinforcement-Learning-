import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def main(num_episodes=1000, render=False, is_training=True): 
    
    env = gym.make("FrozenLake-v1", map_name="4x4", render_mode="human" if render else None, is_slippery=True)
    
    if is_training:
        q_table = np.zeros((env.observation_space.n, env.action_space.n)) # inizialize Q-table 
    else: 
        f = open("q_table.pkl", "rb")
        q_table = pickle.load(f)
        f.close()
        
    learning_rate = 0.1   # alpha --- learning rate
    discount_factor = 0.99 # gamma --- discount factor
    epsilon = 1 # exploration rate. 1 = fully random actions
    epsilon_decay = 0.00005 # decay rate for epsilon
    rng = np.random.default_rng()  # Random number generator
    rewards_per_episode = np.zeros(num_episodes)  # Array to store rewards per episode
    
    for episode in range(num_episodes):
        
        terminated = False
        truncated = False 
        state = env.reset()[0]   
        episode_reward = 0
        while (not terminated) and (not truncated):
            
            if rng.random() < epsilon and is_training:  # Explore: choose a random action
                action = env.action_space.sample()
            else:  # Exploit: choose the best action from Q-table
                action = np.argmax(q_table[state, :])
            
            new_state,reward,terminated,truncated,_ = env.step(action)
            if is_training:
                q_table[state,action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state,:]) 
                                                                              - q_table[state, action]) # formula di aggiornamento Q-table
            value = q_table[state, action]
            state = new_state
            episode_reward += reward
            
        epsilon = max(epsilon - epsilon_decay, 0)  # Decay epsilon after each episode
        if epsilon < 0.01:
            learning_rate = 0.0001  # Reduce learning rate if epsilon is very low
        rewards_per_episode[episode] = episode_reward
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {epsilon:.4f}")
    
    env.close()
    sum_rewards = np.zeros(num_episodes)
    
    for i in range(num_episodes):
        sum_rewards[i] = np.sum(rewards_per_episode[max(0,i-100):i+1])
     
    plt.plot(sum_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards per Episode')
    plt.show()
    if is_training: 
        f = open("q_table.pkl", "wb")
        pickle.dump(q_table, f)
        f.close()
 
       
if __name__ == "__main__":
    main(100000, is_training=False, render=True)  # Set is_training to True for training mode
    
        
