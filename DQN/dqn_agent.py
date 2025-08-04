import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn 
from dqn import DQN 
import random
import flappy_bird_gymnasium
import pickle
import os
from experience_replay import ReplayMemory
import yaml 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    
    def __init__(self, hyperparameters_set):
        with open('DQN/hyperparameters.yml','r') as file:
            all_hyperparameters_set = yaml.safe_load(file) 
            hyperparameters = all_hyperparameters_set[hyperparameters_set]
        
        self.replay_memory_size = hyperparameters["replay_memory_size"]
        self.mini_batch_size    = hyperparameters["mini_batch_size"]
        self.epsilon_init       = hyperparameters["epsilon_init"]
        self.epsilon_decay      = hyperparameters["epsilon_decay"]
        self.epsilon_min        = hyperparameters["epsilon_min"]
        self.network_sync_rate  = hyperparameters["network_sync_rate"]
        self.learning_rate_a    = hyperparameters["learning_rate_a"]
        self.discount_factor_g  = hyperparameters["discount_factor_g"]
        self.num_episodes       = hyperparameters["num_episodes"]
        self.env_id             = hyperparameters["env_id"]
        self.max_reward         = hyperparameters["max_reward"]
        self.max_steps          = hyperparameters["max_steps"]  
        
        self.loss_fcn = nn.MSELoss() 
        
    def optimize(self,mini_batch, policy_dqn, target_dqn):
        
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        
    
        states = torch.stack(states)
        states.requires_grad = True
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)
                
        with torch.no_grad():
            
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
            
            ''' Bellman Equation : Q_target = reward + discount_factor * max(Q_new_state) * (1-done)
            
                target_dqn(new_states) ==> tensor([[1,2,3],[4,5,6]]) --> output: tensor di Q-values per ogni azione possibile 
                .max(1)                ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([2,2]))
                [0]                    ==> tensor([3,6])   --> massimi Q-values per ogni azione possibile
            '''
            
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()            
        ''' Calcolo del current Q-value
            policy_dqn(states) → Q-values per tutte le azioni negli stati correnti
            actions.unsqueeze(1) -> converte le azioni in formato colonna per gather 
            gather(1, ...) -> selezione solo i Q-Values delle azioni effettivamente prese 
            squeeze() -> rimuove la dimensione colonna per avere lo stesso shape di target_q
            
            Esempio: 
             Se actions = [0, 2] e policy_dqn(states) = [[1,2,3], [4,5,6]]
             gather seleziona: [1, 6] (azione 0 dal primo stato, azione 2 dal secondo)
        
        '''

        loss = self.loss_fcn(current_q, target_q)
        policy_dqn.train()  # Prima di chiamare optimize()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
        
    def run(self, is_training=True):
        
        epsilon = self.epsilon_init
        env = gym.make(self.env_id,render_mode="human" if not is_training else None)
        
        obs_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        
        
        episode_rewards = []
        
        if is_training:
            policy_dqn = DQN(obs_space, action_space).to(device) 
            replay_memory = ReplayMemory(10000) 
            target_dqn = DQN(obs_space, action_space).to(device) #creo la target network solo se sto facendo training
            target_dqn.load_state_dict(policy_dqn.state_dict()) #rendo la target uguale alla policy network
            update_count = 0 
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
        else: 
            f = open("nets/" + self.env_id +"_dqn.pkl", "rb")
            policy_dqn = pickle.load(f)
            f.close()
            
        real_episodes = 0
            
        for episode in range(self.num_episodes):
            state,_ = env.reset() 
            state = torch.tensor(state, dtype=torch.float, device=device)
            episode_reward = 0
            terminated = False 
            step_count = 0
            
            while not terminated and step_count <= self.max_steps:
                if is_training and random.random() < epsilon: 
                    action = env.action_space.sample()   # Azione scelta casualmente 
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():  
                        action = policy_dqn(state.unsqueeze(0)).squeeze().argmax()  # Migliore azione suggerita dalla polcy network
                 
                new_state, reward, terminated, truncated, info = env.step(action.item())
                episode_reward += reward
                new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
                reward = torch.tensor(reward, dtype=torch.float32).to(device)
                
                if is_training:
                    replay_memory.append((state,action, new_state, reward, terminated))
                    update_count += 1 
                
                state = new_state
                step_count += 1
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            state,_ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            episode_rewards.append(episode_reward)
            real_episodes += 1
            
            if is_training:
                if  len(replay_memory) > self.mini_batch_size: 
                    mini_batch = replay_memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)
                    if update_count  % self.network_sync_rate == 0: 
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        update_count = 0 
                
            print(f"Episode: {episode}, Steps: {step_count} Reward: {episode_reward}, Epsilon: {epsilon}")
            if episode_reward >= self.max_reward and is_training:
                print("Hai vinto! Complimenti!")
                break

        env.close()
        if is_training: 
            f = open("nets/" + self.env_id +"_dqn.pkl", "wb")
            pickle.dump(policy_dqn, f)
            f.close()
            
            sum_rewards = np.zeros(real_episodes)
        
            for i in range(real_episodes):
                sum_rewards[i] = np.sum(episode_rewards[max(0,i-100):i+1])
                
            plt.plot(sum_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Rewards per Episode')
            plt.show() 
        

if __name__ == "__main__": 
    print(f"CUDA disponibile: {torch.cuda.is_available()}")
    agent = DQNAgent('flappybird') #'flappybird'; 'cartpole'; 'acrobot'
    agent.run(is_training=True)