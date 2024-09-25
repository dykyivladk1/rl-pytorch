import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import collections

# Hyperparameters
LR = 0.0002
GAMMA = 0.98
ROLLOUT = 10
EPISODES = 10000
PRINT_EVERY = 20

class ActorCriticNet(nn.Module):
    def __init__(self, state_size=4, hidden_size=256, action_size=2):
        super(ActorCriticNet, self).__init__()
        self.fc = nn.Linear(state_size, hidden_size)
        self.policy = nn.Linear(hidden_size, action_size)
        self.value = nn.Linear(hidden_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.memory = []

    def forward_policy(self, state):
        x = F.relu(self.fc(state))
        probs = F.softmax(self.policy(x), dim=1)
        return probs

    def forward_value(self, state):
        x = F.relu(self.fc(state))
        val = self.value(x)
        return val

    def store_transition(self, transition):
        self.memory.append(transition)

    def create_batch(self):
        states, actions, rewards, next_states, dones = zip(*self.memory)
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor([0.0 if done else 1.0 for done in dones], dtype=torch.float).unsqueeze(1)
        self.memory = []
        return states, actions, rewards, next_states, dones

    def update(self):
        states, actions, rewards, next_states, dones = self.create_batch()
        
        # Calculate target values
        target_values = rewards + GAMMA * self.forward_value(next_states) * dones
        # Calculate current values
        current_values = self.forward_value(states)
        # Calculate advantage
        advantage = target_values.detach() - current_values

        # Calculate policy loss
        probs = self.forward_policy(states)
        dist = Categorical(probs)
        action_probs = dist.log_prob(actions.squeeze())
        policy_loss = -action_probs * advantage.squeeze()

        # Calculate value loss
        value_loss = F.smooth_l1_loss(current_values, target_values.detach())

        # Total loss
        loss = policy_loss.mean() + value_loss.mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train():
    env = gym.make('CartPole-v1')
    model = ActorCriticNet()
    total_reward = 0

    for episode in range(1, EPISODES + 1):
        state, _ = env.reset()
        done = False

        while not done:
            for _ in range(ROLLOUT):
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                probs = model.forward_policy(state_tensor)
                dist = Categorical(probs)
                action = dist.sample().item()

                next_state, reward, done, truncated, _ = env.step(action)
                model.store_transition((state, action, reward, next_state, done))

                state = next_state
                total_reward += reward

                if done:
                    break

            model.update()

        if episode % PRINT_EVERY == 0:
            avg_reward = total_reward / PRINT_EVERY
            print(f"Episode: {episode}, Average Reward: {avg_reward:.1f}")
            total_reward = 0

    env.close()

if __name__ == "__main__":
    train()
