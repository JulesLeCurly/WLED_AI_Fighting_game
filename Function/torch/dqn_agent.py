import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class DQN(nn.Module):
    """Improved Deep Q-Network with better architecture"""
    
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
    
    def sample(self, batch_size, alpha=0.6):
        """Sample with prioritization"""
        if len(self.buffer) == len(self.priorities):
            priorities = np.array(self.priorities, dtype=np.float32)
        else:
            priorities = np.ones(len(self.buffer), dtype=np.float32)
        
        priorities = priorities ** alpha
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices
        )
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD error"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Fixed DQN Agent - Single Action Selection"""
    
    def __init__(self, state_size, action_size, learning_rate=0.0003, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        # Improved epsilon schedule
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # Higher minimum for continued exploration
        self.epsilon_decay = 0.9995  # Slower decay
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer with gradient clipping
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=50000)
        self.batch_size = 128  # Larger batch for stability
        
        # Training stats
        self.training_step = 0
        self.losses = []
    
    def select_action(self, state, training=True):
        """Select SINGLE action using epsilon-greedy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        """Train the network with proper DQN update"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones, indices = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values: Q(s, a) for the actions taken
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q-values: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Update priorities
        td_errors = (current_q_values - target_q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        self.training_step += 1
        self.losses.append(loss.item())
        
        return loss.item()
    
    def update_target_network(self):
        """Soft update of target network"""
        tau = 0.005  # Soft update parameter
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)
    
    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.training_step = checkpoint.get('training_step', 0)
        print(f"Model loaded from {filepath}")
        print(f"Epsilon: {self.epsilon:.4f}, Training steps: {self.training_step}")