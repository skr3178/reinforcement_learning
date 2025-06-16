import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
from collections import deque

# Hyperparameters
ACTOR_LR = 0.0001
CRITIC_LR = 0.001
GAMMA = 0.99
TAU = 0.001
MEMORY_SIZE = 100000
BATCH_SIZE = 64
MAX_EPISODES = 1000

# Environment setup
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Actor Network
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        # return self.network(state) * max_action
        return self.network(state) 
    # This is a common practice in DDPG to bound the network's output to the action space
    # It allows the network to learn in a normalized space (-1 to 1). 
    # It makes the training more stable as the network doesn't have to learn the exact scale of the actions

# Critic Network
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
    
    def forward(self, state, action):
        return self.network(T.cat([state, action], dim=1))

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)

# Initialize networks
# there are 2 networks for the actor and critic each.
# the actor has a target network which is updated less frequently than the main network.
# actor target & critic target is denoted by Q' and pi'
# actor & critic networks regular is denoted by Q and pi
actor = Actor()
actor_target = Actor()
critic = Critic()
critic_target = Critic()

# Copy parameters to target networks
actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

# Initialize optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)

# Initialize replay buffer
replay_buffer = ReplayBuffer(MEMORY_SIZE)

# Training loop
for episode in range(MAX_EPISODES):
    state = env.reset()
    episode_reward = 0
    done = False

    while not done:
        # Select action
        state_tensor = T.FloatTensor(state)
        action = actor(state_tensor).detach().numpy()
        action = action + np.random.normal(0, 0.1, size=action_dim)  # Exploration noise
        action = np.clip(action, -max_action, max_action) # clip the action to the action space. There is a limit in gym for each env


        # Take action
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        # Store transition in replay buffer
        replay_buffer.push(state, action, reward, next_state, done)

        # Update networks if enough samples
        if len(replay_buffer) > BATCH_SIZE:
            # Sample from replay buffer
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
            
            # Convert to tensors
            states = T.FloatTensor(states)
            actions = T.FloatTensor(actions)
            rewards = T.FloatTensor(rewards).reshape(-1, 1)
            next_states = T.FloatTensor(next_states)
            dones = T.FloatTensor(dones).reshape(-1, 1)

            # Update critic
            # as shown in the pseudocode the y_i is dependent on the pi' network (target actor network).
            next_actions = actor_target(next_states)
            # as shown in the pseudocode the target Q is determined by the reward plus the gamma times the target Q of the next state
            target_Q = critic_target(next_states, next_actions)
            target_Q = rewards + (1 - dones) * GAMMA * target_Q 
            current_Q = critic(states, actions)
            critic_loss = F.mse_loss(current_Q, target_Q.detach())

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Update actor and critic functions, not target actor and critic networks
            actor_loss = -critic(states, actor(states)).mean()
            
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        state = next_state

    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {episode_reward}")

env.close() 