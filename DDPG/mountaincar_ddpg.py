import gym
import numpy as np
import torch
from DDPG import DDPG
from utils import ReplayBuffer
import matplotlib.pyplot as plt
import os
import sys

def train_mountaincar():
    try:
        # Environment setup
        env = gym.make('MountainCarContinuous-v0')
        state_dim = env.observation_space.shape[0]  # 2 for position and velocity
        action_dim = env.action_space.shape[0]      # 1 for continuous force
        action_bounds = env.action_space.high[0]    # Maximum force magnitude
        action_offset = 0                           # No offset needed for this env
        
        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")
        print(f"Action bounds: {action_bounds}")
        
        # Hyperparameters - Reduced for faster training
        max_episodes = 10  # Reduced from 1000
        max_steps = 200     # Reduced from 1000
        batch_size = 32     # Reduced from 64
        n_iter = 5          # Reduced from 10
        lr = 0.001
        H = 200            # Reduced from 1000
        
        # Early stopping parameters
        patience = 20      # Number of episodes to wait for improvement
        min_improvement = 1.0  # Minimum improvement in average reward
        best_avg_reward = float('-inf')
        no_improvement_count = 0
        
        # Initialize DDPG agent
        agent = DDPG(state_dim, action_dim, action_bounds, action_offset, lr, H)
        
        # Initialize replay buffer
        buffer = ReplayBuffer(max_size=1e5)  # Reduced buffer size
        
        # Create models directory if it doesn't exist
        os.makedirs("./models", exist_ok=True)
        
        # Training loop
        episode_rewards = []
        
        for episode in range(max_episodes):
            try:
                state = env.reset()
                episode_reward = 0
                
                for step in range(max_steps):
                    # Select action
                    action = agent.select_action(state, state)
                    
                    # Take action
                    next_state, reward, done, info = env.step(action)
                    
                    # Store transition
                    buffer.add((state, action, reward, next_state, state, 0.99, done))
                    
                    # Update state and reward
                    state = next_state
                    episode_reward += reward
                    
                    # Update policy
                    if buffer.size > batch_size:
                        agent.update(buffer, n_iter, batch_size)
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
                
                # Print progress more frequently
                if (episode + 1) % 5 == 0:  # Changed from 10 to 5
                    avg_reward = np.mean(episode_rewards[-5:])  # Changed from 10 to 5
                    print(f"Episode {episode+1}/{max_episodes}, Average Reward: {avg_reward:.2f}")
                    
                    # Early stopping check
                    if avg_reward > best_avg_reward + min_improvement:
                        best_avg_reward = avg_reward
                        no_improvement_count = 0
                        # Save best model
                        agent.save("./models", "mountaincar_ddpg_best")
                    else:
                        no_improvement_count += 1
                    
                    if no_improvement_count >= patience:
                        print(f"\nNo improvement for {patience} episodes. Stopping training...")
                        break
                    
                    # Save model if performance is good
                    if avg_reward > 90:  # MountainCarContinuous is considered solved at 90
                        print("Solved! Saving model...")
                        agent.save("./models", "mountaincar_ddpg")
                        break
                        
            except KeyboardInterrupt:
                print("\nTraining interrupted by user. Saving current model...")
                agent.save("./models", "mountaincar_ddpg_interrupted")
                break
            except Exception as e:
                print(f"Error during episode {episode}: {str(e)}")
                continue
        
        # Plot training progress
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards)
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig('mountaincar_training.png')
        plt.close()
        
        env.close()
        return agent, episode_rewards
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None, None

if __name__ == "__main__":
    print("Starting MountainCar DDPG training...")
    agent, rewards = train_mountaincar()
    if agent is not None:
        print("Training completed successfully!")
    else:
        print("Training failed!")
        sys.exit(1) 