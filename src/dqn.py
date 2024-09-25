import gym
import collections
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

LR = 0.0005
GAMMA = 0.98
MEMORY_SIZE = 50000
BATCH = 32
EPISODES = 10000
PRINT_EVERY = 20

class Memory:
    def __init__(self):
        self.store = collections.deque(maxlen=MEMORY_SIZE)
    
    def add(self, experience):
        self.store.append(experience)
    
    def sample(self, size):
        batch = random.sample(self.store, size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.tensor(states, dtype=torch.float),
            torch.tensor(actions).unsqueeze(1),
            torch.tensor(rewards).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float),
            torch.tensor(dones).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.store)

class QNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden=128, output_dim=2):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
    
    def choose_action(self, state, eps):
        if random.random() < eps:
            return random.randint(0, 1)
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax().item()

def train_model(policy, target, memory, optimizer):
    for _ in range(10):
        states, actions, rewards, next_states, dones = memory.sample(BATCH)
        
        current_q = policy(states).gather(1, actions)
        max_next_q = target(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + GAMMA * max_next_q * dones
        
        loss = F.smooth_l1_loss(current_q, target_q)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env = gym.make('CartPole-v1')
    policy_net = QNetwork()
    target_net = QNetwork()
    target_net.load_state_dict(policy_net.state_dict())
    memory = Memory()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    total_reward = 0
    
    for episode in range(1, EPISODES + 1):
        eps = max(0.01, 0.08 - 0.01 * (episode / 200))
        state, _ = env.reset()
        done = False
        
        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            action = policy_net.choose_action(state_tensor, eps)
            next_state, reward, done, _, _ = env.step(action)
            done_flag = 0.0 if done else 1.0
            
            memory.add((state, action, reward / 100.0, next_state, done_flag))
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        if len(memory) > 2000:
            train_model(policy_net, target_net, memory, optimizer)
        
        if episode % PRINT_EVERY == 0:
            target_net.load_state_dict(policy_net.state_dict())
            avg_reward = total_reward / PRINT_EVERY
            print(f"Episode: {episode}, Avg Reward: {avg_reward:.1f}, Memory: {len(memory)}, Epsilon: {eps*100:.1f}%")
            total_reward = 0
    
    env.close()

if __name__ == "__main__":
    main()
