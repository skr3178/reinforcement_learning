"""
A2C Example - Simple demonstration of the A2C algorithm on the CartPole environment

This script shows how to use the A2C implementation to solve the CartPole environment.
It serves as a practical example to understand how A2C works.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from a2c import A2C, train

# Set parameters
ENV_NAME = "CartPole-v1"
NUM_EPISODES = 500
MAX_STEPS = 500
GAMMA = 0.99
LEARNING_RATE = 0.001
RENDER = False  # Set to True to visualize training (slows down training)

def main():
    print(f"Training A2C agent on {ENV_NAME} environment")
    print(f"Parameters: Episodes={NUM_EPISODES}, Max Steps={MAX_STEPS}, Gamma={GAMMA}, LR={LEARNING_RATE}")
    
    # Train the agent
    agent, rewards = train(
        env_name=ENV_NAME,
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        gamma=GAMMA,
        lr=LEARNING_RATE,
        render=RENDER
    )
    
    # Print final performance
    avg_reward = np.mean(rewards[-10:])
    print(f"\nTraining completed!")
    print(f"Final average reward over last 10 episodes: {avg_reward:.2f}")
    
    # Plot learning curve
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'A2C Learning Curve - {ENV_NAME}')
        plt.savefig('a2c_learning_curve.png')
        plt.show()
    except ImportError:
        print("Matplotlib not installed, skipping plot")
    
    # Test the trained agent
    test_agent(agent, ENV_NAME, num_episodes=5, render=True)

def test_agent(agent, env_name, num_episodes=5, render=True):
    """
    Test the trained agent on the environment
    
    Args:
        agent: Trained A2C agent
        env_name: Name of the environment
        num_episodes: Number of test episodes
        render: Whether to render the environment
    """
    print("\nTesting the trained agent...")
    
    # Create environment
    env = gym.make(env_name, render_mode="human" if render else None)
    
    test_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            action, _ = agent.network.get_action(state)
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            # Update state and episode reward
            state = next_state
            episode_reward += reward
        
        test_rewards.append(episode_reward)
        print(f"Test Episode {episode+1}: Reward = {episode_reward}")
    
    env.close()
    print(f"Average test reward: {np.mean(test_rewards):.2f}")

if __name__ == "__main__":
    main()