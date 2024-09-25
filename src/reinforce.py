import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

lr = 0.0002
g = 0.98

class Reinforce(nn.Module):
    def __init__(self):
        super(Reinforce, self).__init__()

        self.mem = []
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def store(self, item):
        self.mem.append(item)

    def learn(self):
        R = 0
        self.opt.zero_grad()
        for rew, prob in self.mem[::-1]:
            # reverse the rewards 
            # the cumulative return at each time step need to know the future returns
            R = rew + g * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.opt.step()
        self.mem = []

def main():
    env = gym.make('CartPole-v1')
    pi = Reinforce()
    score = 0.0
    print_int = 20

    for ep in range(10000):
        s, _ = env.reset()
        done = False
        
        while not done:
            probs = pi(torch.from_numpy(s).float())
            act_dist = Categorical(probs)
            act = act_dist.sample()
            s_n, r, done, trunc, info = env.step(act.item())
            pi.store((r, probs[act]))
            s = s_n
            score += r

        pi.learn()

        if ep % print_int == 0 and ep != 0:
            print("# Ep: {}, avg score: {}".format(ep, score/print_int))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()
