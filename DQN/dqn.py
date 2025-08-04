import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

if __name__ == "__main__":
    
    state_dim = 12  # Example state dimension (e.g., CartPole)
    action_dim = 2  # Example action dimension (e.g., left/right)
    model = DQN(state_dim, action_dim)
    state = torch.randn(10, state_dim)  # Example state
    q_values = model(state)
    print("Q-values:", q_values)

        