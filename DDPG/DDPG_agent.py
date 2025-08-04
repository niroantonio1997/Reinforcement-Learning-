import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn 
import random
import os
import yaml 
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utilities.network import net
from Utilities.replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent: 
    
    def __init__(self, hyperparameters_set):
        
        with open('DDPG/hyperparameters.yml','r') as file:
            all_hyperparameters_set = yaml.safe_load(file) 
            hyperparameters = all_hyperparameters_set[hyperparameters_set]
        
        self.gamma = hyperparameters["gamma"]
        self.tau = hyperparameters["tau"]
        self.buffer_size = hyperparameters["buffer_size"]
        self.batch_size = hyperparameters["batch_size"]
        self.exploration_noise_std = hyperparameters["exploration_noise_std"]
        self.train_freq = hyperparameters["train_freq"]
        self.max_episode_steps = hyperparameters["max_episode_steps"]
        self.total_timesteps = hyperparameters["total_timesteps"]
        self.env_id = hyperparameters["env_id"]
        self.max_reward = hyperparameters["max_reward"]
        self.max_steps = hyperparameters["max_episode_steps"]
        
        # Parametri per exploration noise decay
        self.noise_std = self.exploration_noise_std
        self.noise_decay = 0.995
        self.min_noise = 0.01
        
        
    def run(self, is_training):
        
        env = gym.make(self.env_id, render_mode="human" if not is_training else None)
        obs_space = env.observation_space.shape[0]
        
        action_dim = env.action_space.shape[0]
        action_low  = env.action_space.low[0]  # Per Pendulum è scalare
        action_high = env.action_space.high[0]
        action_scale = (action_high - action_low) / 2.0
        action_bias  = (action_high + action_low) / 2.0

        episodes_reward = []
        if is_training:
            actor_hidden_dims  = [128, 128]
            critic_hidden_dims = [128, 128]
            actor_lr  = 5e-5
            critic_lr = 1e-4 
            actor = net(obs_space, actor_hidden_dims, action_dim, output_activiation=torch.tanh).to(device)
            critic = net(obs_space + action_dim, critic_hidden_dims, 1).to(device)
            actor_target = net(obs_space, actor_hidden_dims, action_dim, output_activiation=torch.tanh).to(device)
            critic_target = net(obs_space + action_dim, critic_hidden_dims, 1).to(device)
            
            actor_target.load_state_dict(actor.state_dict())
            critic_target.load_state_dict(critic.state_dict())
            
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
            
            replay_buffer = ReplayBuffer(obs_space, action_dim, self.buffer_size)
            episode = 0
            episode_reward = 0
            
            obs, _ = env.reset()
            
            for t in range(self.total_timesteps):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                
                # Select action - CORREZIONE CRITICA
                if replay_buffer.size < 1000:  # Solo i primi 1000 step random
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        raw_action = actor(obs_tensor).cpu().numpy()[0]  # Rimuovi batch dimension
                        # Applica scaling correttamente
                        action = raw_action * action_scale + action_bias
                        # Aggiungi noise per exploration
                        noise = np.random.normal(0, self.noise_std, size=action.shape)
                        action = np.clip(action + noise, action_low, action_high)
                
                # Take step
                next_obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                
                # Store in replay buffer
                replay_buffer.store(obs, action, reward, next_obs, terminated)
                
                # Train - CORREZIONE: train più frequentemente
                if replay_buffer.size >= self.batch_size and t % self.train_freq == 0:
                    mini_batch = replay_buffer.sample(self.batch_size)
                    self.optimize(mini_batch, actor, actor_target, critic, critic_target, 
                                actor_optimizer, critic_optimizer)
                
                obs = next_obs
                
                if terminated or truncated:
                    obs, _ = env.reset()
                    episode += 1
                    print(f"Episode {episode}, Reward: {episode_reward:.2f}, Steps: {t}, Noise: {self.noise_std:.3f}")
                    episodes_reward.append(episode_reward)
                    episode_reward = 0
                    
                    # Decay noise
                    self.noise_std = max(self.min_noise, self.noise_std * self.noise_decay)
                        
            # Plotting
            if episodes_reward:
                # Smooth rewards con finestra mobile
                window_size = min(100, len(episodes_reward))
                smoothed_rewards = []
                for i in range(len(episodes_reward)):
                    start_idx = max(0, i - window_size + 1)
                    smoothed_rewards.append(np.mean(episodes_reward[start_idx:i+1]))
                
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.plot(episodes_reward, alpha=0.3, label='Raw')
                plt.plot(smoothed_rewards, label=f'Smoothed ({window_size})')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.title('Training Progress')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(1, 2, 2)
                plt.plot(smoothed_rewards[-200:])  # Ultimi 200 episodi
                plt.xlabel('Episode (last 200)')
                plt.ylabel('Smoothed Reward')
                plt.title('Recent Performance')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            
            # Save networks
            os.makedirs("DDPG/nets/", exist_ok=True)
            torch.save(actor.state_dict(), f"DDPG/nets/{self.env_id}_actor.pth")
            torch.save(critic.state_dict(), f"DDPG/nets/{self.env_id}_critic.pth")
            
        else: 
            # Inference mode - CORREZIONE
            actor = net(obs_space, actor_hidden_dims, action_dim, output_activiation=torch.tanh).to(device)
            actor.load_state_dict(torch.load(f"DDPG/nets/{self.env_id}_actor.pth"))
            actor.eval()
            
            obs, _ = env.reset()
            episode_reward = 0
            terminated = False
            
            while not terminated:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    raw_action = actor(obs_tensor).cpu().numpy()[0]
                    action = raw_action * action_scale + action_bias
                    action = np.clip(action, action_low, action_high)
                
                next_obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                obs = next_obs
                
                if truncated:
                    terminated = True
                    
            print(f"Test Episode Reward: {episode_reward:.2f}")
                
    
    def optimize(self, mini_batch, actor, actor_target, critic, critic_target, actor_optimizer, critic_optimizer):
        
        states = torch.FloatTensor(mini_batch["states"]).to(device)
        actions = torch.FloatTensor(mini_batch["acts"]).to(device)
        new_states = torch.FloatTensor(mini_batch["new_states"]).to(device)
        rewards = torch.FloatTensor(mini_batch["rews"]).to(device).unsqueeze(1)
        terminations = torch.FloatTensor(mini_batch["done"]).to(device).unsqueeze(1)
        
        # Critic update
        with torch.no_grad():
            next_actions = actor_target(new_states)
            q_target = rewards + (1 - terminations) * self.gamma * critic_target(torch.cat((new_states, next_actions), dim=1))
        
        q = critic(torch.cat((states, actions), dim=1))
        critic_loss = nn.MSELoss()(q, q_target)
        
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Actor update
        actor_loss = -critic(torch.cat([states, actor(states)], dim=1)).mean() 
        
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
        # Soft update target networks
        for param, target_param in zip(actor.parameters(), actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


if __name__ == "__main__":
    agent = DDPGAgent('walker')
    agent.run(is_training=True)