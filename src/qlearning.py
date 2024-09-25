import random
import numpy as np

class Environment:
    def __init__(self, rows=5, cols=7, goal=(4, 6)):
        self.rows = rows
        self.cols = cols
        self.goal = goal
        self.reset()

    def step(self, action):
        if action == 'left':
            self._move(direction='left')
        elif action == 'up':
            self._move(direction='up')
        elif action == 'right':
            self._move(direction='right')
        elif action == 'down':
            self._move(direction='down')
        
        reward = -1
        done = self._is_goal_reached()
        return (self.position_x, self.position_y), reward, done

    def _move(self, direction):
        if direction == 'left' and self.position_y > 0:
            self.position_y -= 1
        elif direction == 'right' and self.position_y < self.cols - 1:
            self.position_y += 1
        elif direction == 'up' and self.position_x > 0:
            self.position_x -= 1
        elif direction == 'down' and self.position_x < self.rows - 1:
            self.position_x += 1

    def _is_goal_reached(self):
        return (self.position_x, self.position_y) == self.goal

    def reset(self):
        self.position_x = 0
        self.position_y = 0
        return (self.position_x, self.position_y)


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.9, min_exploration=0.2, exploration_decay=0.01):
        self.q_table = np.zeros((env.rows, env.cols, 4))  # Actions: left, up, right, down
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.min_epsilon = min_exploration
        self.epsilon_decay = exploration_decay
        self.action_space = ['left', 'up', 'right', 'down']

    def choose_action(self, state):
        x, y = state
        if random.random() < self.epsilon:
            return random.randint(0, 3) 
        else:
            return np.argmax(self.q_table[x, y])  

    def learn(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        predict = self.q_table[x, y, action]
        target = reward + self.gamma * np.max(self.q_table[next_x, next_y])
        self.q_table[x, y, action] += self.lr * (target - predict)

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)

    def display_policy(self, env):
        policy_grid = np.full((env.rows, env.cols), ' ', dtype=str)
        for i in range(env.rows):
            for j in range(env.cols):
                best_action = np.argmax(self.q_table[i, j])
                policy_grid[i, j] = self.action_space[best_action][0].upper()
        policy_grid[env.goal[0], env.goal[1]] = 'G' 
        print(policy_grid)


def run_training(episodes=1000):
    env = Environment()
    agent = QLearningAgent(env)

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action_index = agent.choose_action(state)
            action = agent.action_space[action_index]
            next_state, reward, done = env.step(action)
            agent.learn(state, action_index, reward, next_state)
            state = next_state

        agent.update_epsilon()

    agent.display_policy(env)


if __name__ == '__main__':
    run_training()
