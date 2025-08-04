<<<<<<< HEAD
# 🤖 Reinforcement Learning Implementations

A comprehensive collection of reinforcement learning algorithms implemented in PyTorch, designed for various continuous and discrete control environments.

## 📋 Algorithms Implemented

### PPO (Proximal Policy Optimization)
- **Features**: GAE (Generalized Advantage Estimation), entropy regularization, adaptive learning rate
- **Environments**: Humanoid, Ant, Hopper, Walker2d, HalfCheetah, CartPole, LunarLander, etc.
- **Special Features**: 
  - Ctrl+C safe interruption with auto-save
  - Dynamic buffer size adjustment
  - Comprehensive training plots and statistics

### DQN (Deep Q-Network)
- **Features**: Experience replay, target network, epsilon-greedy exploration
- **Environments**: CartPole, Acrobot, discrete control tasks

### DDPG (Deep Deterministic Policy Gradient) 
- **Features**: Actor-Critic architecture, Ornstein-Uhlenbeck noise
- **Environments**: Continuous control tasks (Pendulum, BipedalWalker)

## 🚀 Quick Start

### Installation
```bash
python -m venv rl_env
source rl_env/bin/activate  # On Windows: rl_env\Scripts\activate
pip install torch gymnasium[mujoco] matplotlib pyyaml numpy
```

### Training an Agent
```bash
cd PPO
python PPO_agent.py
```

### Testing a Trained Agent
Set `is_training = False` in the main function and run the script.

## 📁 Project Structure
```
Reinforcement_Learning/
├── PPO/
│   ├── PPO_agent.py           # Main PPO implementation
│   ├── hyperparameters.yml   # Environment-specific configs
│   └── nets/                  # Saved models
├── DQN/
│   ├── dqn_agent.py          # DQN implementation
│   └── nets/                  # Saved models
├── DDPG/
│   ├── DDPG_agent.py         # DDPG implementation
│   └── nets/                  # Saved models
└── Utilities/
    ├── network.py            # Neural network architectures
    └── replay_buffer.py      # Experience replay buffer
```

## 🎮 Supported Environments

### MuJoCo Environments
- **Humanoid-v5**: 17-DOF humanoid locomotion
- **Ant-v4**: Quadrupedal ant locomotion  
- **Hopper-v5**: 2D hopping robot
- **Walker2d-v4**: 2D walking robot
- **HalfCheetah-v4**: Running cheetah
- **Pendulum-v1**: Inverted pendulum control

### Gymnasium Classic Control
- **CartPole-v1**: Cart pole balancing
- **LunarLander-v3**: Lunar landing simulation
- **MountainCar**: Continuous mountain car

## 🔧 Key Features

### Intelligent Training Management
- **Safe Interruption**: Press Ctrl+C to safely stop training and save progress
- **Auto-saving**: Models and training data automatically saved
- **Dynamic Parameters**: Buffer size increases during training for better stability
- **Comprehensive Logging**: Detailed statistics and performance plots

### Advanced PPO Implementation
- **GAE (λ=0.95)**: Reduces variance in advantage estimation
- **Entropy Regularization**: Maintains exploration throughout training
- **Learning Rate Scheduling**: Exponential decay for fine-tuning
- **Gradient Clipping**: Prevents exploding gradients
- **Batch Normalization**: Stabilizes advantage estimates

### MuJoCo Camera Control
- **Tracking Camera**: Automatically follows the agent
- **Customizable Views**: Adjustable distance, elevation, and azimuth

## 📊 Hyperparameter Configuration

All hyperparameters are centralized in `hyperparameters.yml`:

```yaml
humanoid:
  env_id: Humanoid-v5
  gamma: 0.99
  buffer_size: 16000
  train_freq: 50
  actor_lr: 1e-3
  critic_lr: 3e-3
  entropy_coef: 0.1
  std: 0.1
  minibatch_size: 16000
```

## 📈 Performance Monitoring

### Training Plots
- Real-time reward progression
- Smoothed performance curves  
- Statistical summaries
- Recent performance analysis

### Training Statistics
- Episodes completed
- Maximum/mean rewards
- Standard deviation
- Convergence indicators

## 🎯 Benchmark Results

| Environment | Algorithm | Mean Reward | Training Episodes |
|-------------|-----------|-------------|-------------------|
| CartPole-v1 | PPO | 475+ | ~1,000 |
| Pendulum-v1 | PPO | -200+ | ~500 |
| Hopper-v5 | PPO | 2000+ | ~5,000 |
| Ant-v4 | PPO | 4000+ | ~10,000 |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI Gymnasium for the environments
- MuJoCo for physics simulation
- PyTorch team for the deep learning framework
- PPO paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

## 📞 Contact

For questions or suggestions, feel free to open an issue or contact the maintainer.

---

⭐ **Star this repository if you find it helpful!** ⭐
