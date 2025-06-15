"""
A2C (Advantage Actor-Critic) Reinforcement Learning Algorithm Implementation

This file implements a simple version of the A2C algorithm to help understand its core concepts.
A2C is an on-policy algorithm that combines:
1. An actor (policy network) that decides which actions to take
2. A critic (value network) that evaluates how good those actions are

Key concepts:
- Actor-Critic: Combines policy-based (actor) and value-based (critic) methods
- Advantage: Uses the advantage function A(s,a) = Q(s,a) - V(s) to reduce variance
- On-policy: Uses current policy to collect experiences (unlike off-policy methods)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    """
    Combined actor-critic network with shared feature extraction layers.

    The actor outputs a probability distribution over actions (policy).
    The critic outputs a value estimate for the current state.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()

        # Shared feature extraction layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            # No softmax here as we'll use log_softmax in the forward pass
        )

        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """Forward pass through the network"""
        x = self.shared(x)

        # Actor: output action probabilities
        action_probs = F.softmax(self.actor(x), dim=-1)

        # Critic: output state value
        state_value = self.critic(x)

        return action_probs, state_value

    def get_action(self, state):
        """Sample an action from the policy distribution"""
        # is this the same state as defined from env.reset and after executing an action in the env.step?
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs, _ = self.forward(state)

        # Sample from the action probability distribution
        action_dist = torch.distributions.Categorical(action_probs)
        # stochastic nature of the sampling
        action = action_dist.sample()

        # Return action and log probability
        return action.item(), action_dist.log_prob(action)

class A2C:
    """
    Advantage Actor-Critic (A2C) agent implementation
    """
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.001):
        """
        Initialize the A2C agent
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            gamma: Discount factor for future rewards
            lr: Learning rate
        """
        self.gamma = gamma

        # Create the actor-critic network
        self.network = ActorCritic(state_dim, action_dim).to(device)

        # Create optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def update(self, states, actions, rewards, next_states, dones):
        """
        Update the actor-critic network using collected experiences
        Args:
            states: Batch of states
            actions: Batch of actions taken
            rewards: Batch of rewards received
            next_states: Batch of next states
            dones: Batch of done flags (True if episode ended after this step)
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Get action probabilities and state values|| invokes the forward method of the network
        action_probs, state_values = self.network(states)

        # Get next state values for bootstrapping
        _, next_state_values = self.network(next_states)
        next_state_values = next_state_values.squeeze()

        # Compute target values using TD(0) for the critic value function
        # If done, next_state_value = 0
        # Qsa[s,a] = Qsa[s,a] + alpha * (r + gamma * np.max(Qsa[s_next])-Qsa[s,a])
        # including dones ensure that If done=True (episode ends), (1 - dones) = 0, so: target_values = rewards + 0 = rewards
        # target_values = rewards + gamma * next_state_values
        target_values = rewards + self.gamma * next_state_values * (1 - dones)

        # Compute advantage: A(s,a) = R + γV(s') - V(s)
        # If A(s,a) > 0, the action was better than average.
        # If A(s,a) < 0, the action was worse.
        advantages = target_values - state_values.squeeze()

        # Get log probabilities of actions taken
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)

        # Compute losses
        # Actor loss: -log(π(a|s)) * A(s,a)
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Critic loss: MSE between predicted value and target value
        critic_loss = F.mse_loss(state_values.squeeze(), target_values.detach())

        # Entropy bonus for exploration
        # entropy(): High entropy = more exploration (actions have similar probabilities).
        # Low entropy = policy is deterministic (risks getting stuck in suboptimal actions).
        # Intuition: Including entropy prevents premature convergence to bad policies.

        entropy = action_dist.entropy().mean()

        # Total loss (minus entropy to encourage exploration)
        total_loss = actor_loss + critic_loss - 0.01 * entropy

        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

def train(env_name, num_episodes=1000, max_steps=1000, gamma=0.99, lr=0.001, render=False):
    """
    Train the A2C agent on the specified environment

    Args:
        env_name: Name of the Gym environment
        num_episodes: Number of episodes to train for
        max_steps: Maximum steps per episode
        gamma: Discount factor
        lr: Learning rate
        render: Whether to render the environment
    """
    # Create environment
    env = gym.make(env_name)

    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]

    # Check if action space is discrete
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    else:
        raise ValueError("This implementation only supports discrete action spaces")

    # Create A2C agent
    agent = A2C(state_dim, action_dim, gamma, lr)

    # Training loop
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        # Lists to store episode experiences
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for step in range(max_steps):
            if render:
                env.render()

            # Select action
            action, _ = agent.network.get_action(state)

            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(float(done))

            # Update state and episode reward
            state = next_state
            episode_reward += reward

            if done:
                break

        # Update agent
        loss = agent.update(states, actions, rewards, next_states, dones)

        # Track progress
        episode_rewards.append(episode_reward)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")

    env.close()
    return agent, episode_rewards

if __name__ == "__main__":
    # Train on CartPole environment
    agent, rewards = train("CartPole-v1", num_episodes=500, render=False)

    # Print final performance
    print(f"Final average reward over last 10 episodes: {np.mean(rewards[-10:]):.2f}")

    # Plot learning curve
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('A2C Learning Curve')
        plt.savefig('a2c_learning_curve.png')
        plt.show()
    except ImportError:
        print("Matplotlib not installed, skipping plot")
