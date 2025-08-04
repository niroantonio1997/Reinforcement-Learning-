# DDPG (Deep Deterministic Policy Gradient) Implementation

## Features

### Advanced Training Features
- **Video Recording**: Automatically saves MP4 videos every 10% of training episodes
- **Safe Interruption**: Press Ctrl+C to safely save models and plot results
- **Best Model Tracking**: Automatically saves the best performing models
- **Early Stopping**: Training stops when target average reward is reached
- **Universal Rendering**: Compatible with both MuJoCo and non-MuJoCo environments

### Technical Features
- **Soft Target Updates**: Stabilized training with target networks
- **Experience Replay**: Efficient learning from past experiences
- **Exploration Noise Decay**: Adaptive exploration strategy
- **Gradient Clipping**: Prevents gradient explosion
- **Comprehensive Logging**: Detailed training statistics and plots

## Usage

### Training
```python
from DDPG_agent import DDPGAgent
import os

# Examples for different environments:

# Simple environments (fast training)
hyperparameters_set = "pendulum"         # Classic control
hyperparameters_set = "invertedpendulum" # Simple MuJoCo

# Medium complexity 
hyperparameters_set = "walker"           # BipedalWalker
hyperparameters_set = "hopper"           # MuJoCo hopper
hyperparameters_set = "reacher"          # MuJoCo arm

# High complexity (long training)
hyperparameters_set = "ant"              # Quadruped
hyperparameters_set = "halfcheetah"      # Fast runner
hyperparameters_set = "humanoid"         # Most complex

# Create directories
os.makedirs(f'DDPG/video/{hyperparameters_set}', exist_ok=True)
os.makedirs('DDPG/graphs', exist_ok=True)

# Train agent
agent = DDPGAgent(hyperparameters_set)
agent.run(is_training=True)
```

### Testing
```python
agent = DDPGAgent(hyperparameters_set)
agent.run(is_training=False)
```

## Supported Environments

### Classic Control
- **Pendulum-v1**: Classic inverted pendulum control (Target: -150)

### Box2D 
- **BipedalWalker-v3**: Humanoid walking challenge (Target: 300)

### MuJoCo Environments
- **Ant-v4**: Quadruped ant locomotion (Target: 6000)
- **HalfCheetah-v4**: Fast running cheetah (Target: 4000)
- **Hopper-v4**: One-legged hopping robot (Target: 3500)
- **Walker2d-v4**: Two-legged walking robot (Target: 4000)
- **Humanoid-v4**: Complex humanoid locomotion (Target: 6000)
- **Swimmer-v4**: Swimming in 2D (Target: 100)
- **InvertedPendulum-v4**: Simple balance task (Target: 950)
- **InvertedDoublePendulum-v4**: Double pendulum balance (Target: 9000)
- **Reacher-v4**: Arm reaching task (Target: -5)
- **Pusher-v4**: Object manipulation task (Target: -20)

## Hyperparameters

Configure training in `hyperparameters.yml`:
- `max_average_reward`: Target performance threshold
- `actor_lr`/`critic_lr`: Learning rates for networks
- `gamma`: Discount factor
- `tau`: Soft update rate for target networks
- `exploration_noise_std`: Initial exploration noise

### Environment-Specific Optimizations

**Simple Environments** (pendulum, invertedpendulum):
- Smaller networks (128x128)
- Higher learning rates (0.001)
- Less training time

**Medium Complexity** (hopper, walker2d, reacher):
- Medium networks (256x256)
- Moderate learning rates (0.0001)
- Balanced exploration noise

**Complex Environments** (ant, halfcheetah, humanoid):
- Large networks (256x256 or 512x512)
- Lower learning rates (0.00005-0.0001)
- Large replay buffers
- Extended training time

## Output

- **Models**: Saved in `DDPG/nets/`
- **Videos**: Saved in `DDPG/video/{environment}/`
- **Graphs**: Training plots in `DDPG/graphs/`
- **Training Data**: Backup saved on interruption

## Key Improvements over Standard DDPG

1. **Stability**: Better action scaling and target network updates
2. **Monitoring**: Real-time video recording and comprehensive plots
3. **Robustness**: Safe interruption and automatic best model saving
4. **Compatibility**: Universal rendering system for different environments
