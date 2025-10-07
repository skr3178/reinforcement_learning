import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

# Hyper-parameters
seed = 1
render = False
num_episodes = 2000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
env = gym.make('CartPole-v0').unwrapped
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
torch.manual_seed(seed)
np.random.seed(seed)

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.fc2 = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_value = self.fc2(x)
        return action_value

class DQN():

    capacity = 8000
    warmup_size = 1000  # Start training after this many samples
    learning_rate = 1e-3
    memory_count = 0
    batch_size = 256
    gamma = 0.995
    update_count = 0
    epsilon = 1.0  # Start with full exploration
    epsilon_min = 0.01
    epsilon_decay = 0.995

    def __init__(self):
        super(DQN, self).__init__()
        self.target_net, self.act_net = Net().to(device), Net().to(device)
        self.memory = [None]*self.capacity
        self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.writer = SummaryWriter('./DQN/logs')


    def select_action(self,state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
        value = self.act_net(state)
        action_max_value, index = torch.max(value, 1)
        action = index.item()
        if np.random.rand(1) < self.epsilon: # epsilon greedy with decay
            action = np.random.choice(range(num_action), 1).item()
        return action

    def store_transition(self,transition):
        index = self.memory_count % self.capacity
        self.memory[index] = transition
        self.memory_count += 1
        return self.memory_count >= self.capacity

    def update(self):
        if self.memory_count >= self.warmup_size:
            # Get valid memory size
            memory_size = min(self.memory_count, self.capacity)
            
            # Sample ONE random minibatch
            indices = np.random.choice(memory_size, self.batch_size, replace=False)
            batch = [self.memory[i] for i in indices]
            
            # Convert to tensors efficiently
            state = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float).to(device)
            action = torch.LongTensor([t.action for t in batch]).view(-1,1).to(device)
            reward = torch.tensor([t.reward for t in batch], dtype=torch.float).to(device)
            next_state = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float).to(device)
            done = torch.tensor([t.done for t in batch], dtype=torch.float).to(device)

            # Bellman equation: Q(s,a) = r + γ * max Q(s',a') for non-terminal states
            # For terminal states (done=True), Q(s,a) = r only (no future value!)
            with torch.no_grad():
                next_q_values = self.target_net(next_state).max(1)[0]
                target_v = reward + self.gamma * next_q_values * (1 - done)

            # Single gradient update
            current_v = self.act_net(state).gather(1, action).squeeze()
            loss = self.loss_func(current_v, target_v)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.writer.add_scalar('loss/value_loss', loss, self.update_count)
            self.update_count += 1
            
            if self.update_count % 100 == 0:
                self.target_net.load_state_dict(self.act_net.state_dict())
            
            return loss.item()
        return None
def main():

    agent = DQN()
    episode_steps = []
    solved_threshold = 195.0  # CartPole-v0 is considered solved at 195 average reward
    
    pbar = tqdm(range(num_episodes), desc="Training DQN")
    for i_ep in pbar:
        state, _ = env.reset(seed=seed if i_ep == 0 else None)
        if render: env.render()
        for t in range(10000):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if render: env.render()
            transition = Transition(state, action, reward, next_state, done)
            agent.store_transition(transition)
            
            # Update network every step (not just at episode end)
            loss = agent.update()
            
            state = next_state
            if done or t >=9999:
                agent.writer.add_scalar('live/finish_step', t+1, global_step=i_ep)
                episode_steps.append(t+1)
                
                # Decay epsilon after each episode
                agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
                
                # Update progress bar
                avg_steps = np.mean(episode_steps[-100:]) if len(episode_steps) > 0 else 0
                pbar.set_postfix({'steps': t+1, 'avg_100': f'{avg_steps:.1f}', 'eps': f'{agent.epsilon:.3f}'})
                
                # Check if solved (need at least 100 episodes)
                if len(episode_steps) >= 100 and avg_steps >= solved_threshold:
                    print(f"\n🎉 Environment SOLVED in {i_ep+1} episodes!")
                    print(f"Average score over last 100 episodes: {avg_steps:.2f} >= {solved_threshold}")
                    pbar.close()
                    return
                
                break
    
    # Training completed all episodes
    final_avg = np.mean(episode_steps[-100:]) if len(episode_steps) >= 100 else np.mean(episode_steps)
    print(f"\nTraining complete! Final average (last 100 episodes): {final_avg:.2f}")
    if final_avg >= solved_threshold:
        print(f"✓ Environment SOLVED!")
    else:
        print(f"✗ Not solved. Gap: {solved_threshold - final_avg:.2f} steps")

if __name__ == '__main__':
    main()