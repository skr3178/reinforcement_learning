import math
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]  # 4 for CartPole
action_dim = env.action_space.n  # 2 for CartPole

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

actor = Actor()
critic = Critic()

actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

num_episodes = 1000
gamma = 0.99

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    done = False

    while not done:
        # Convert state to tensor
        state_tensor = T.FloatTensor(state)
        
        # Get action probabilities from actor
        action_probs = actor(state_tensor)
        action = T.multinomial(action_probs, 1).item()

        # Take action and observe next state
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        # Convert next state to tensor
        next_state_tensor = T.FloatTensor(next_state)

        # Compute advantage
        state_value = critic(state_tensor)
        next_state_value = critic(next_state_tensor)
        advantage = reward + gamma * next_state_value * (1 - done) - state_value

        # Update critic
        critic_loss = advantage.pow(2).mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Update actor
        actor_loss = -(T.log(action_probs[action]) * advantage.detach()).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        state = next_state

    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {episode_reward}")

env.close()