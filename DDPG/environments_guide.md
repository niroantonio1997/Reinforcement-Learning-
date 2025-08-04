# DDPG Environments Guide

## Quick Reference

### 🚀 Fast Training (< 30 min)
```python
hyperparameters_set = "pendulum"              # Classic pendulum
hyperparameters_set = "invertedpendulum"      # Simple balance
```

### ⚡ Medium Training (1-3 hours)
```python
hyperparameters_set = "walker"                # BipedalWalker
hyperparameters_set = "hopper"                # One-legged robot
hyperparameters_set = "walker2d"              # Two-legged robot
hyperparameters_set = "reacher"               # Arm reaching
hyperparameters_set = "swimmer"               # Swimming task
hyperparameters_set = "inverteddoublependulum"# Double pendulum
```

### 🔥 Long Training (3+ hours)
```python
hyperparameters_set = "ant"                   # Quadruped ant
hyperparameters_set = "halfcheetah"           # Fast cheetah
hyperparameters_set = "humanoid"              # Complex humanoid
hyperparameters_set = "pusher"                # Object manipulation
```

## Environment Details

| Environment | Difficulty | Target Reward | Training Time | Description |
|-------------|------------|---------------|---------------|-------------|
| **pendulum** | ⭐ | -150 | ~20 min | Classic inverted pendulum |
| **invertedpendulum** | ⭐ | 950 | ~15 min | Simple balance task |
| **walker** | ⭐⭐ | 300 | ~2 hours | BipedalWalker locomotion |
| **hopper** | ⭐⭐ | 3500 | ~1 hour | One-legged hopping |
| **walker2d** | ⭐⭐ | 4000 | ~2 hours | Two-legged walking |
| **reacher** | ⭐⭐ | -5 | ~1 hour | Arm reaching target |
| **swimmer** | ⭐⭐ | 100 | ~1.5 hours | Swimming locomotion |
| **inverteddoublependulum** | ⭐⭐ | 9000 | ~1 hour | Double pendulum balance |
| **ant** | ⭐⭐⭐ | 6000 | ~4 hours | Quadruped locomotion |
| **halfcheetah** | ⭐⭐⭐ | 4000 | ~3 hours | Fast running |
| **humanoid** | ⭐⭐⭐⭐ | 6000 | ~8 hours | Complex humanoid |
| **pusher** | ⭐⭐⭐ | -20 | ~5 hours | Object manipulation |

## Network Architectures

### Simple Tasks
- **Networks**: 128×128
- **Learning Rate**: 0.001
- **Buffer Size**: 100K-500K

### Medium Tasks  
- **Networks**: 256×256
- **Learning Rate**: 0.0001
- **Buffer Size**: 500K-1M

### Complex Tasks
- **Networks**: 256×256 or 512×512×256
- **Learning Rate**: 0.00005-0.0001
- **Buffer Size**: 1M-2M

## Usage Example

```python
from DDPG_agent import DDPGAgent
import os

# Select environment (change this line)
env_name = "hopper"  # Try: pendulum, ant, humanoid, etc.

# Setup and train
os.makedirs(f'DDPG/video/{env_name}', exist_ok=True)
os.makedirs('DDPG/graphs', exist_ok=True)

agent = DDPGAgent(env_name)
agent.run(is_training=True)
```

## Expected Outputs

All environments will generate:
- **Videos**: `DDPG/video/{env_name}/episode_X_{env_id}.mp4`
- **Models**: `DDPG/nets/{env_id}_actor.pth`, `DDPG/nets/{env_id}_critic.pth`
- **Graphs**: `DDPG/graphs/training_plot_{env_id}.png`
- **Data**: Training progress saved on Ctrl+C interrupt
