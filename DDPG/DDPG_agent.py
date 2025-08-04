import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn 
import random
import os
import yaml 
import sys 
import signal
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utilities.network import net
from Utilities.replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent: 
    
    def __init__(self, hyperparameters_set):
        
        with open('DDPG/hyperparameters.yml','r') as file:
            all_hyperparameters_set = yaml.safe_load(file) 
            hyperparameters = all_hyperparameters_set[hyperparameters_set]
        
        self.hyperparameters_set = hyperparameters_set  # Salva il nome del set per i path
        self.gamma = hyperparameters["gamma"]
        self.tau = hyperparameters["tau"]
        self.buffer_size = hyperparameters["buffer_size"]
        self.batch_size = hyperparameters["batch_size"]
        self.exploration_noise_std = hyperparameters["exploration_noise_std"]
        self.train_freq = hyperparameters["train_freq"]
        self.max_episode_steps = hyperparameters["max_episode_steps"]
        self.total_timesteps = hyperparameters["total_timesteps"]
        self.max_episodes = hyperparameters["max_episodes"]
        self.env_id = hyperparameters["env_id"]
        self.max_average_reward = hyperparameters["max_average_reward"]
        self.actor_hidden_dims = hyperparameters["actor_hidden_dims"]
        self.critic_hidden_dims = hyperparameters["critic_hidden_dims"]
        self.actor_lr = float(hyperparameters["actor_lr"])
        self.critic_lr = float(hyperparameters["critic_lr"])
        
        # Parametri per exploration noise decay
        self.noise_std = self.exploration_noise_std
        self.noise_decay = 0.995
        self.min_noise = 0.01
        
        # Variables for interrupt handling
        self.episodes_reward = []
        self.interrupted = False
        
    def is_mujoco_env(self, env_id):
        """
        Check if the environment is a MuJoCo environment
        """
        mujoco_envs = [
            'Humanoid', 'Ant', 'Hopper', 'Walker2d', 'HalfCheetah', 
            'InvertedPendulum', 'InvertedDoublePendulum', 'Swimmer',
            'Reacher', 'Pusher', 'Thrower', 'Striker'
        ]
        return any(env_name in env_id for env_name in mujoco_envs) or 'mujoco' in env_id.lower()
    
    def save_episode_video(self, frames, episode_num):
        """
        Save a sequence of frames as a video
        """
        if len(frames) > 0:
            import imageio
            filename = f'DDPG/video/{self.hyperparameters_set}/episode_{episode_num}_{self.env_id}.mp4'
            # Salva come video MP4 con 30 fps
            imageio.mimsave(filename, frames, fps=30, codec='libx264')
            print(f"Episode video saved as {filename}")
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C interrupt"""
        print("\n\n=== TRAINING INTERRUPTED ===")
        print("Saving models and plotting results...")
        self.interrupted = True
        
        # Save models if they exist
        if hasattr(self, 'best_actor') and hasattr(self, 'best_critic'):
            torch.save(self.best_actor.state_dict(), f'DDPG/nets/{self.env_id}_actor.pth')
            torch.save(self.best_critic.state_dict(), f'DDPG/nets/{self.env_id}_critic.pth')

        # Save training data
        with open(f'DDPG/training_data_{self.env_id}_interrupted.pkl', 'wb') as f:
            pickle.dump({
                'episodes_reward': self.episodes_reward,
                'env_id': self.env_id,
                'interrupted_at_episode': len(self.episodes_reward)
            }, f)
        print(f"Training data saved as training_data_{self.env_id}_interrupted.pkl")
        
        # Plot results
        if len(self.episodes_reward) > 0:
            self.plot_results(self.episodes_reward, interrupted=True)
        
        print("=== CLEANUP COMPLETE ===")
        sys.exit(0)
        
        
    def run(self, is_training):

        if is_training:
            # Crea l'ambiente con parametri appropriati
            if self.is_mujoco_env(self.env_id):
                # Ambiente MuJoCo - usa parametri camera
                env = gym.make(self.env_id, render_mode="rgb_array", camera_name="track", width=1200, height=800)
            else:
                # Ambiente non-MuJoCo - usa solo render_mode
                env = gym.make(self.env_id, render_mode="rgb_array")

            obs_space = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            
            # Setup action scaling per DDPG
            self.action_low = torch.tensor(env.action_space.low, dtype=torch.float32).to(device)
            self.action_high = torch.tensor(env.action_space.high, dtype=torch.float32).to(device)
            action_scale = (self.action_high - self.action_low) / 2.0
            action_bias = (self.action_high + self.action_low) / 2.0
            
            # Setup signal handler for Ctrl+C
            signal.signal(signal.SIGINT, self.signal_handler)
            print("=== DDPG TRAINING STARTED ===")
            print("Press Ctrl+C to safely interrupt and save progress")
            print("=" * 50)
            
            # Initialize networks
            self.actor = net(obs_space, self.actor_hidden_dims, action_dim, output_activiation=torch.tanh).to(device)
            self.critic = net(obs_space + action_dim, self.critic_hidden_dims, 1).to(device)
            self.actor_target = net(obs_space, self.actor_hidden_dims, action_dim, output_activiation=torch.tanh).to(device)
            self.critic_target = net(obs_space + action_dim, self.critic_hidden_dims, 1).to(device)
            
            # Initialize best models for tracking
            self.best_actor = net(obs_space, self.actor_hidden_dims, action_dim, output_activiation=torch.tanh).to(device)
            self.best_critic = net(obs_space + action_dim, self.critic_hidden_dims, 1).to(device)
            
            # Copy initial weights to targets and best models
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.best_actor.load_state_dict(self.actor.state_dict())
            self.best_critic.load_state_dict(self.critic.state_dict())
            
            # Initialize optimizers
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
            
            # Initialize replay buffer
            self.replay_buffer = ReplayBuffer(obs_space, action_dim, self.buffer_size, 'continuous')
            
            self.episodes_reward = []  # Initialize here for signal handler
            self.best_reward = -np.inf
            episode = 0
            total_steps = 0
            
            for episode in range(self.max_episodes):
                render_flag = False 
                save_video = False
                frames = []  # Lista per salvare i frame del video
                
                if (episode) % (self.max_episodes // 10) == 0:
                    save_video = True  # Abilita il rendering ogni 10% degli episodi
                
                # Reset environment
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                steps = 0
                
                while not done and steps < self.max_episode_steps:
                    steps += 1
                    total_steps += 1
                    
                    # Select action
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    
                    if self.replay_buffer.size < 1000:  # Random exploration iniziale
                        action = env.action_space.sample()
                    else:
                        with torch.no_grad():
                            raw_action = self.actor(obs_tensor).cpu().numpy()[0]
                            # Applica scaling: da [-1,1] a [action_low, action_high]
                            action = raw_action * action_scale.cpu().numpy() + action_bias.cpu().numpy()
                            # Aggiungi noise per exploration
                            noise = np.random.normal(0, self.noise_std, size=action.shape)
                            action = np.clip(action + noise, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())
                    
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    
                    # Cattura frame per video se necessario
                    if save_video:
                        rgb_array = env.render()
                        if rgb_array is not None:
                            frames.append(rgb_array)
                    
                    done = terminated or truncated
                    
                    # Store experience
                    self.replay_buffer.store(obs, action, reward, next_obs, done, 0.0)  # log_prob=0 per DDPG
                    
                    # Train
                    if self.replay_buffer.size >= self.batch_size and total_steps % self.train_freq == 0:
                        mini_batch = self.replay_buffer.sample(self.batch_size)
                        self.optimize(mini_batch)
                    
                    obs = next_obs
                    episode_reward += reward
                    
                # Track best model
                if episode_reward >= self.best_reward:
                    self.best_reward = episode_reward
                    self.best_actor.load_state_dict(self.actor.state_dict())
                    self.best_critic.load_state_dict(self.critic.state_dict())
                
                # Salva video se necessario
                if save_video and len(frames) > 0:
                    self.save_episode_video(frames, episode)
                
                self.episodes_reward.append(episode_reward)
                
                # Decay noise
                self.noise_std = max(self.min_noise, self.noise_std * self.noise_decay)
                
                # Check if solved
                if len(self.episodes_reward) >= 100 and np.mean(self.episodes_reward[-100:]) >= self.max_average_reward:
                    print(f"Solved in {episode+1} episodes with average reward {np.mean(self.episodes_reward[-100:]):.2f}!")
                    break
                
                # Debug info ogni 10 episodi
                if (episode + 1) % 10 == 0:
                    avg_reward = np.mean(self.episodes_reward[-10:])
                    video_info = " [VIDEO SAVED]" if save_video else ""
                    print(f"Episode {episode+1}/{self.max_episodes}, Avg Reward (last 10): {avg_reward:.2f}, Current: {episode_reward:.2f}, Steps: {steps}, Noise: {self.noise_std:.4f}{video_info}")
                else:
                    video_info = " [VIDEO SAVED]" if save_video else ""
                    print(f"Episode {episode+1}/{self.max_episodes}, Reward: {episode_reward:.2f}, Steps: {steps}{video_info}")
                
                # Check for interrupt
                if self.interrupted:
                    break
                
            env.close()
            
            # Save best models
            torch.save(self.best_actor.state_dict(), f'DDPG/nets/{self.env_id}_actor.pth')
            torch.save(self.best_critic.state_dict(), f'DDPG/nets/{self.env_id}_critic.pth')

            # Plot results
            self.plot_results(self.episodes_reward)
            
        else: # TEST
            # Crea l'ambiente con parametri appropriati per il test
            if self.is_mujoco_env(self.env_id):
                # Ambiente MuJoCo - usa parametri camera
                env = gym.make(self.env_id, render_mode="human", camera_name="track", width=1200, height=800)
            else:
                # Ambiente non-MuJoCo - usa solo render_mode
                env = gym.make(self.env_id, render_mode="human")
                
            obs_space = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            
            # Setup action scaling
            self.action_low = torch.tensor(env.action_space.low, dtype=torch.float32).to(device)
            self.action_high = torch.tensor(env.action_space.high, dtype=torch.float32).to(device)
            action_scale = (self.action_high - self.action_low) / 2.0
            action_bias = (self.action_high + self.action_low) / 2.0
            
            # Load trained model
            self.actor = net(obs_space, self.actor_hidden_dims, action_dim, output_activiation=torch.tanh).to(device)
            self.actor.load_state_dict(torch.load(f'DDPG/nets/{self.env_id}_actor.pth'))
            self.actor.eval()

            while True:
                obs, _ = env.reset()
                env.render()
                
                done = False
                episode_reward = 0
                steps = 0
                    
                while not done and steps < self.max_episode_steps:
                    steps += 1
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    
                    with torch.no_grad():
                        # Use deterministic policy for testing (no noise)
                        raw_action = self.actor(obs_tensor).cpu().numpy()[0]
                        action = raw_action * action_scale.cpu().numpy() + action_bias.cpu().numpy()
                        action = np.clip(action, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())
        
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    obs = next_obs
                    episode_reward += reward 
                    
                print(f"Test Episode Reward: {episode_reward:.2f}, Steps: {steps}")
                
    
    def optimize(self, mini_batch):
        """
        Optimize actor and critic networks using DDPG
        """
        states = torch.FloatTensor(mini_batch["states"]).to(device)
        actions = torch.FloatTensor(mini_batch["acts"]).to(device)
        new_states = torch.FloatTensor(mini_batch["new_states"]).to(device)
        rewards = torch.FloatTensor(mini_batch["rews"]).to(device).unsqueeze(1)
        terminations = torch.FloatTensor(mini_batch["done"]).to(device).unsqueeze(1)
        
        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(new_states)
            q_target = rewards + (1 - terminations) * self.gamma * self.critic_target(torch.cat((new_states, next_actions), dim=1))
        
        q = self.critic(torch.cat((states, actions), dim=1))
        critic_loss = nn.MSELoss()(q, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(torch.cat([states, self.actor(states)], dim=1)).mean() 
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        
        # Soft update target networks
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def plot_results(self, episodes_reward, interrupted=False):
        """
        Plot training results
        """
        if len(episodes_reward) == 0:
            print("No episodes completed yet, cannot plot results.")
            return
            
        window_size = min(100, len(episodes_reward))
        smoothed_rewards = []
        for i in range(len(episodes_reward)):
            start_idx = max(0, i - window_size + 1)
            smoothed_rewards.append(np.mean(episodes_reward[start_idx:i+1]))
        
        plt.figure(figsize=(15, 5))
        
        # Full training progress
        plt.subplot(1, 3, 1)
        plt.plot(episodes_reward, alpha=0.3, label='Raw', color='lightblue')
        plt.plot(smoothed_rewards, label=f'Smoothed ({window_size})', color='blue')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        title = f'DDPG Training Progress - {self.env_id}'
        if interrupted:
            title += f' (INTERRUPTED at episode {len(episodes_reward)})'
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        # Recent performance
        plt.subplot(1, 3, 2)
        recent_episodes = min(200, len(episodes_reward))
        plt.plot(smoothed_rewards[-recent_episodes:], color='green')
        plt.xlabel(f'Episode (last {recent_episodes})')
        plt.ylabel('Smoothed Reward')
        plt.title('Recent Performance')
        plt.grid(True)
        
        # Statistics
        plt.subplot(1, 3, 3)
        stats_text = f"""
            DDPG Training Statistics:
            Episodes completed: {len(episodes_reward)}
            Max reward: {max(episodes_reward):.2f}
            Mean reward: {np.mean(episodes_reward):.2f}
            Last 100 mean: {np.mean(episodes_reward[-100:]):.2f}
            Std deviation: {np.std(episodes_reward):.2f}

            Target reward: {self.max_average_reward}
            Environment: {self.env_id}
            """
        if interrupted:
            stats_text += "\nSTATUS: INTERRUPTED"
        else:
            stats_text += "\nSTATUS: COMPLETED"
            
        plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        filename = f'DDPG/graphs/training_plot_{self.env_id}'
        if interrupted:
            filename += '_interrupted'
        filename += '.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")
        
        plt.show()


def main():
    # Choose your environment:
    # Simple/Fast: "pendulum", "invertedpendulum"  
    # Medium: "walker", "hopper", "walker2d", "reacher"
    # Complex: "ant", "halfcheetah", "humanoid", "swimmer", "pusher"
    
    hyperparameters_set = "pendulum"  # Change this to desired environment
    
    os.makedirs('DDPG/video/' + hyperparameters_set, exist_ok=True)
    os.makedirs('DDPG/graphs', exist_ok=True)
    agent = DDPGAgent(hyperparameters_set)
    
    is_training = True
    agent.run(is_training)  


if __name__ == "__main__":
    main()