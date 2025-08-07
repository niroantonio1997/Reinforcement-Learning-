import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn 
import random
import flappy_bird_gymnasium
import os
import yaml 
import sys
import signal
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utilities.network import net
from Utilities.replay_buffer import ReplayBuffer
from Utilities.common_functions import is_mujoco_env, save_episode_video, create_signal_handler, plot_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOAgent: 
    
    def __init__(self, hyperparameters_set):
        
        with open('PPO/hyperparameters.yml','rb') as file:
            all_hyperparameters_set = yaml.safe_load(file) 
            hyperparameters = all_hyperparameters_set[hyperparameters_set]
        
        self.hyperparameters_set = hyperparameters_set  # Salva il nome del set per i path
        self.gamma = hyperparameters["gamma"]
        self.lam = hyperparameters.get("lambda", 0.95)  # GAE lambda parameter
        self.buffer_size = hyperparameters["buffer_size"]
        self.train_freq = hyperparameters["train_freq"]
        self.max_episode_steps = hyperparameters["max_episode_steps"]
        self.max_episodes = hyperparameters["max_episodes"]
        self.total_timesteps = hyperparameters["total_timesteps"]
        self.env_id = hyperparameters["env_id"]
        self.action_type = hyperparameters["action_type"]
        self.actor_hidden_dims  = hyperparameters["actor_hidden_dims"]
        self.critic_hidden_dims = hyperparameters["critic_hidden_dims"]
        self.actor_lr  = float(hyperparameters["actor_lr"])
        self.critic_lr = float(hyperparameters["critic_lr"])
        self.entropy_coef = hyperparameters.get("entropy_coef", 0.01)
        self.entropy_coef_decay = hyperparameters.get("entropy_coef_decay", 0.99)
        self.epsilon = hyperparameters.get("epsilon", 0.2)  # Clip range for PPO
        self.minibatch_size = self.buffer_size
        self.max_average_reward = hyperparameters["max_average_reward"]
        
        # Per azioni continue
        self.std = hyperparameters.get("std", 0.1)
        self.std_decay = hyperparameters.get("std_decay", 0.99)
        
        # Variables for interrupt handling
        self.episodes_reward = []
        self.interrupted = False

    #calcola il Generalized Advantage Estimation (GAE)
    def compute_gae(self, rewards, dones, values, next_value=0):
        """
        Compute Generalized Advantage Estimation (GAE)
        """
        advantages = []
        gae = 0
        
        # Aggiungi il valore bootstrap finale
        values = np.append(values, next_value)
        
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages.insert(0, gae)
            
        return advantages

    #Calcola i returns, ovvero la somma scontata dei rewards
    def compute_returns(self, rewards, dones, next_value=0):
        """
        Compute discounted returns
        """
        returns = []
        G = next_value
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                G = 0
            G = rewards[i] + self.gamma * G
            returns.insert(0, G)

        return returns
    
    #Ottimizza le reti attore e critico usando PPO
    def optimize(self, states, actions, old_log_probs, returns, advantages):
        """
        Optimize actor and critic networks using PPO
        """
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        # Multiple epochs of optimization
        for epoch in range(self.train_freq):
            np.random.shuffle(indices)
            
            # Mini-batch training
            for start in range(0, dataset_size, self.minibatch_size):
                end = start + self.minibatch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Normalize advantages for this batch
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
                
                # Actor loss
                if self.action_type == 'continuous':
                    # Ricalcola mu_scaled per essere consistente con il sampling
                    mu = self.actor(batch_states)
                    mu_scaled = self.action_low + (mu + 1.0) * 0.5 * (self.action_high - self.action_low)
                    std = torch.full_like(mu_scaled, self.std)
                    new_dist = torch.distributions.Normal(mu_scaled, std)
                    new_log_prob = new_dist.log_prob(batch_actions).sum(dim=-1)
                    entropy = new_dist.entropy().sum(dim=-1).mean()
                else:
                    logits = self.actor(batch_states)
                    new_dist = torch.distributions.Categorical(logits=logits)
                    new_log_prob = new_dist.log_prob(batch_actions)
                    entropy = new_dist.entropy().mean()

                ratio = torch.exp(new_log_prob - batch_old_log_probs)
                
                # PPO clipped objective
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                actor_loss = -torch.min(surrogate1, surrogate2).mean() - self.entropy_coef * entropy
                
                # Critic loss
                values = self.critic(batch_states).squeeze(-1)
                critic_loss = nn.MSELoss()(values, batch_returns)
                
                # Update networks
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()
                
            # Step scheduler dopo ogni ottimizzazione
            self.actor_scheduler.step()
            self.critic_scheduler.step()
        
        # Decay entropy coefficient and std
        self.entropy_coef *= self.entropy_coef_decay
        if self.action_type == 'continuous':
            self.std = max(0.01, self.std * self.std_decay)  # Evita che std diventi troppo piccolo

    def update(self):
        """
        Update the agent using collected experiences
        """
        buffer = self.replay_buffer.return_all()
        states = torch.tensor(buffer['states'], dtype=torch.float32).to(device)
        rewards = np.array(buffer['rews'])
        dones = np.array(buffer['done'])
        log_probs = torch.tensor(buffer['log_probs'], dtype=torch.float32).to(device)
        
        if self.action_type == 'continuous':
            actions = torch.tensor(buffer['acts'], dtype=torch.float32).to(device)
            if actions.ndim == 1:
                actions = actions.unsqueeze(-1)
        else: 
            actions = torch.tensor(buffer['acts'], dtype=torch.long).to(device)

        # Compute values for all states
        with torch.no_grad():
            values = self.critic(states).squeeze(-1).cpu().numpy()
        
        # Compute advantages using GAE
        advantages = self.compute_gae(rewards, dones, values)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        
        # Compute returns
        returns = self.compute_returns(rewards, dones)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        
        # Optimize networks
        self.optimize(states, actions, log_probs, returns, advantages)
        
    def run(self, is_training):

        if is_training:
            # Crea l'ambiente con parametri appropriati
            if is_mujoco_env(self.env_id):
                # Ambiente MuJoCo - usa parametri camera
                env = gym.make(self.env_id, render_mode="rgb_array", camera_name="track", width=1200, height=800)
            else:
                # Ambiente non-MuJoCo - usa solo render_mode
                env = gym.make(self.env_id, render_mode="rgb_array")
                
            env._max_episode_steps = self.max_episode_steps  # Imposta il limite di passi per l'episodio

            obs_space = env.observation_space.shape[0]

            if self.action_type == 'continuous':
                action_dim = env.action_space.shape[0]
                self.action_low = torch.tensor(env.action_space.low, dtype=torch.float32).to(device)
                self.action_high = torch.tensor(env.action_space.high, dtype=torch.float32).to(device)
            else:
                action_dim = env.action_space.n
            
            # Setup signal handler for Ctrl+C
            signal.signal(signal.SIGINT, create_signal_handler(self, "PPO"))
            print("=== TRAINING STARTED ===")
            print("Press Ctrl+C to safely interrupt and save progress")
            print("=" * 50)
            
            # Initialize networks
            if self.action_type == 'continuous':
                # Per azioni continue, usiamo tanh come attivazione finale
                self.actor = net(obs_space, self.actor_hidden_dims, action_dim, torch.tanh).to(device)
                self.best_actor = net(obs_space, self.actor_hidden_dims, action_dim, torch.tanh).to(device)  
            else:
                self.actor = net(obs_space, self.actor_hidden_dims, action_dim).to(device)  
                self.best_actor = net(obs_space, self.actor_hidden_dims, action_dim).to(device)
            self.critic = net(obs_space, self.critic_hidden_dims, 1).to(device)
            self.best_critic = net(obs_space, self.critic_hidden_dims, 1).to(device)
            
            # Initialize the replay buffer
            buffer_type = 'continuous' if self.action_type == 'continuous' else 'discrete'
            self.replay_buffer = ReplayBuffer(obs_space, action_dim, self.buffer_size, buffer_type)
            
            # Initialize optimizers
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
            self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.999999995)
            self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.999999995)
            
            self.episodes_reward = []  # Initialize here for signal handler
            total_steps = 0
            self.best_reward = -np.inf
            
            for episode in range(self.max_episodes):
                render_flag = False 
                save_video = False
                frames = []  # Lista per salvare i frame del video
                
                if (episode) % (self.max_episodes // 10) == 0:
                    self.buffer_size = int(self.buffer_size * 1.1)  # Aumenta la dimensione del buffer per gli episodi successivi
                    self.replay_buffer = ReplayBuffer(obs_space, action_dim, self.buffer_size, buffer_type)
                    save_video = True  # Abilita il rendering ogni 10% degli episodi per monitorare i progressi
                # Reset environment
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                steps = 0
                
                while not done and steps < self.max_episode_steps:
                    steps += 1
                    total_steps += 1
                    
                    # Select action
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        if self.action_type == 'continuous':
                            # La rete ora produce output in [-1, 1] grazie al tanh
                            mu = self.actor(obs_tensor)
                            # Scala nell'intervallo dell'ambiente
                            mu_scaled = self.action_low + (mu + 1.0) * 0.5 * (self.action_high - self.action_low)
                            std = torch.full_like(mu_scaled, self.std)
                            dist = torch.distributions.Normal(mu_scaled, std)
                            action_tensor = dist.sample()
                            # Clamp per sicurezza
                            action_tensor = torch.clamp(action_tensor, self.action_low, self.action_high)
                            action = action_tensor.cpu().numpy().flatten().astype(np.float32)
                            log_prob = dist.log_prob(action_tensor).sum(dim=-1).item()
                        else:
                            logits = self.actor(obs_tensor)
                            dist = torch.distributions.Categorical(logits=logits)
                            action = dist.sample().item()
                            log_prob = dist.log_prob(torch.tensor(action).to(device)).item()
       
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    
                    # Cattura frame per video se necessario
                    if save_video:
                        rgb_array = env.render()
                        if rgb_array is not None:
                            frames.append(rgb_array)
                    
                    done = terminated or truncated

                    # Store experience
                    if self.action_type == 'continuous':
                        # Memorizza l'azione sampionata direttamente
                        self.replay_buffer.store(obs, action_tensor.cpu().numpy().flatten(), reward, next_obs, done, log_prob)
                    else:
                        self.replay_buffer.store(obs, action, reward, next_obs, done, log_prob)
                    
                    obs = next_obs
                    episode_reward += reward
                    
                if episode_reward >= self.best_reward:
                    self.best_reward = episode_reward
                    self.best_actor.load_state_dict(self.actor.state_dict())
                    self.best_critic.load_state_dict(self.critic.state_dict())
                
                # Salva video se necessario
                if save_video and len(frames) > 0:
                    save_episode_video(frames, episode,  self.hyperparameters_set, self.env_id, "PPO")
                    
                # Update agent when buffer is full
                if self.replay_buffer.size >= self.buffer_size:
                    print(f"Updating agent at episode {episode+1}...")
                    self.update()
                    print(f"Actor learning rate: {self.actor_optimizer.param_groups[0]['lr']}, critic learning rate: {self.critic_optimizer.param_groups[0]['lr']}")
                    self.replay_buffer.clear()
                    
                self.episodes_reward.append(episode_reward)
                if np.mean(self.episodes_reward[-500:]) >= self.max_average_reward:
                    print(f"Solved in {episode+1} episodes with reward {episode_reward:.2f}!")
                    break
                
                # Debug info ogni 10 episodi per monitoraggio più frequente
                if (episode + 1) % 10 == 0:
                    avg_reward = np.mean(self.episodes_reward[-10:])
                    video_info = " [VIDEO SAVED]" if save_video else ""
                    print(f"Episode {episode+1}/{self.max_episodes}, Avg Reward (last 10): {avg_reward:.2f}, Current: {episode_reward:.2f}, Steps: {steps}, Std: {self.std:.4f}{video_info}")
                else:
                    video_info = " [VIDEO SAVED]" if save_video else ""
                    print(f"Episode {episode+1}/{self.max_episodes}, Reward: {episode_reward:.2f}, Steps: {steps}{video_info}")
                
                # Check for interrupt
                if self.interrupted:
                    break
                
            env.close()
            torch.save(self.best_actor.state_dict(), f'PPO/nets/{self.env_id}_actor.pth')
            torch.save(self.best_critic.state_dict(), f'PPO/nets/{self.env_id}_critic.pth')

            # Plot results
            plot_results(self.episodes_reward, self.env_id, self.max_average_reward, "PPO")            
        
        ################# TEST MODE #################    
        else: 
            # Crea l'ambiente con parametri appropriati per il test
            if is_mujoco_env(self.env_id):
                # Ambiente MuJoCo - usa parametri camera
                env = gym.make(self.env_id, render_mode="rgb_array", camera_name="track", width=1200, height=800)
            else:
                # Ambiente non-MuJoCo - usa solo render_mode
                env = gym.make(self.env_id, render_mode="rgb_array")
                
            env._max_episode_steps = self.max_episode_steps  # Imposta il limite di passi per l'episodio
            obs_space = env.observation_space.shape[0]
            
            action_dim = env.action_space.shape[0] if self.action_type == 'continuous' else env.action_space.n
            
            if self.action_type == 'continuous':
                self.actor = net(obs_space, self.actor_hidden_dims, action_dim, torch.tanh).to(device)
            else:
                self.actor = net(obs_space, self.actor_hidden_dims, action_dim).to(device)
            self.actor.load_state_dict(torch.load(f'PPO/nets/{self.env_id}_actor.pth'))
            self.actor.eval()
            frames = []  # Lista per salvare i frame del video
            obs, _ = env.reset()
            
            if self.action_type == 'continuous':
                action_dim = env.action_space.shape[0]
                self.action_low = torch.tensor(env.action_space.low, dtype=torch.float32).to(device)
                self.action_high = torch.tensor(env.action_space.high, dtype=torch.float32).to(device)
            else:
                action_dim = env.action_space.n                
    
            done = False
            episode_reward = 0
            steps = 0
            episode = 0
            while not done and steps < self.max_episode_steps:
                steps += 1
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    if self.action_type == 'continuous':
                        # Use deterministic policy for testing
                        mu = self.actor(obs_tensor)
                        mu_scaled = self.action_low + (mu + 1.0) * 0.5 * (self.action_high - self.action_low)
                        action = mu_scaled.detach().cpu().numpy().flatten().astype(np.float32)
                    else:
                        logits = self.actor(obs_tensor)
                        action = torch.argmax(logits, dim=-1).item()
    
                next_obs, reward, terminated, truncated, _ = env.step(action)
                rgb_array = env.render()
                if rgb_array is not None:
                    frames.append(rgb_array)
                done = terminated or truncated
                
                obs = next_obs
                episode_reward += reward 
            save_episode_video(frames, None,  self.hyperparameters_set, self.env_id, "PPO")
                
            print(f"Test Episode Reward: {episode_reward:.2f}, Steps: {steps}")

def main():
    hyperparameters_set = "flappybird"  # Change this to the desired hyperparameters set
    os.makedirs('PPO/video/' + hyperparameters_set, exist_ok=True)
    os.makedirs('PPO/graphs', exist_ok=True)
    agent = PPOAgent(hyperparameters_set)
    
    is_training = False  # Set to True for training, False for testing
    agent.run(is_training)  

if __name__ == "__main__":
    main()