import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys


def is_mujoco_env(env_id):
    """
    Check if the environment is a MuJoCo environment
    """
    mujoco_envs = [
        'Humanoid', 'Ant', 'Hopper', 'Walker2d', 'HalfCheetah', 
        'InvertedPendulum', 'InvertedDoublePendulum', 'Swimmer',
        'Reacher', 'Pusher', 'Thrower', 'Striker'
    ]
    return any(env_name in env_id for env_name in mujoco_envs) or 'mujoco' in env_id.lower()


def save_episode_video(frames, episode_num, hyperparameters_set, env_id, algorithm_name):
    """
    Save a sequence of frames as a video
    """
    if len(frames) > 0:
        import imageio
        filename = f'{algorithm_name}/video/{hyperparameters_set}/episode_{episode_num if episode_num is not None else "final"}_{env_id}.mp4'
        # Salva come video MP4 con 30 fps
        imageio.mimsave(filename, frames, fps=30, codec='libx264')
        print(f"Episode video saved as {filename}")


def create_signal_handler(agent, algorithm_name):
    """
    Create a signal handler for Ctrl+C interrupt
    """
    def signal_handler(signum, frame):
        """Handle Ctrl+C interrupt"""
        print(f"\n\n=== {algorithm_name.upper()} TRAINING INTERRUPTED ===")
        print("Saving models and plotting results...")
        agent.interrupted = True
        
        # Save models if they exist
        if hasattr(agent, 'best_actor') and hasattr(agent, 'best_critic'):
            import torch
            torch.save(agent.best_actor.state_dict(), f'{algorithm_name}/nets/{agent.env_id}_actor.pth')
            torch.save(agent.best_critic.state_dict(), f'{algorithm_name}/nets/{agent.env_id}_critic.pth')

        # Save training data
        with open(f'{algorithm_name}/training_data_{agent.env_id}_interrupted.pkl', 'wb') as f:
            pickle.dump({
                'episodes_reward': agent.episodes_reward,
                'env_id': agent.env_id,
                'interrupted_at_episode': len(agent.episodes_reward)
            }, f)
        print(f"Training data saved as training_data_{agent.env_id}_interrupted.pkl")
        
        # Plot results
        if len(agent.episodes_reward) > 0:
            plot_results(agent.episodes_reward, agent.env_id, agent.max_average_reward, algorithm_name, interrupted=True)
        
        print("=== CLEANUP COMPLETE ===")
        sys.exit(0)
    
    return signal_handler


def plot_results(episodes_reward, env_id, max_average_reward, algorithm_name, interrupted=False):
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
    title = f'{algorithm_name.upper()} Training Progress - {env_id}'
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
        {algorithm_name.upper()} Training Statistics:
        Episodes completed: {len(episodes_reward)}
        Max reward: {max(episodes_reward):.2f}
        Mean reward: {np.mean(episodes_reward):.2f}
        Last 100 mean: {np.mean(episodes_reward[-100:]):.2f}
        Std deviation: {np.std(episodes_reward):.2f}

        Target reward: {max_average_reward}
        Environment: {env_id}
        """
    if interrupted:
        stats_text += "\nSTATUS: INTERRUPTED"
    else:
        stats_text += "\nSTATUS: COMPLETED"
        
    plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save plot
    filename = f'{algorithm_name}/graphs/training_plot_{env_id}'
    if interrupted:
        filename += '_interrupted'
    filename += '.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")
    
    plt.show()
