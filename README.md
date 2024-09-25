# Reinforcement Learning Algorithms

This repository contains implementations of key reinforcement learning (RL) algorithms, including Actor-Critic, Deep Q-Networks (DQN), Q-Learning, and REINFORCE. Each algorithm demonstrates different approaches to solving the RL problem of learning from environments and maximizing rewards through interactions.

## Repository Structure

- **`actor-critic.py`**: Implements the **Actor-Critic algorithm**, a hybrid method that combines both policy-based (Actor) and value-based (Critic) approaches. The Actor updates the policy directly, while the Critic estimates the value of the state to guide the Actor's learning.
  
- **`dqn.py`**: Implements **Deep Q-Network (DQN)**, a value-based method that approximates Q-values using deep neural networks. It extends the classic Q-learning algorithm by using experience replay and target networks to stabilize learning.
  
- **`qlearning.py`**: Implements **Q-Learning**, a value-based reinforcement learning algorithm that updates the value of actions in a table (Q-table). This algorithm is the foundation of many modern RL techniques but does not scale well to large state spaces.
  
- **`reinforce.py`**: Implements the **REINFORCE algorithm**, a Monte Carlo policy gradient method. REINFORCE is a policy-based approach that updates the policy by computing gradients of the expected return with respect to the policy parameters, ensuring optimal behavior over time.

## How to Run

To run the scripts:

1. Install the required dependencies (e.g., `gym`, `torch`).
2. Run the individual scripts using Python.
   - Example: `python qlearning.py`

## Dependencies

Ensure you have the following dependencies installed:
- `gym`: OpenAI's toolkit for developing and comparing RL algorithms.
- `torch`: PyTorch, the deep learning framework used for neural network computations.

