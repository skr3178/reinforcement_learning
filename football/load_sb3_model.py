#!/usr/bin/env python3
"""
Script to load and use a trained Stable-Baselines3 PPO model in GFootball.
"""

from stable_baselines3 import PPO
import gfootball.env as football_env
import numpy as np

def load_and_play_with_model(model_path, level="11_vs_11_easy_stochastic", render=True):
    """
    Load a trained PPO model and play with it in GFootball.
    
    Args:
        model_path: Path to the saved model (.zip file)
        level: GFootball scenario level
        render: Whether to render the game
    """
    
    print(f"Loading model from: {model_path}")
    
    # Create the environment with the same settings used during training
    env = football_env.create_environment(
        env_name=level,
        stacked=True,  # Should match training settings
        rewards='scoring,checkpoints',  # Should match training settings
        render=render
    )
    
    # Load the trained model
    model = PPO.load(model_path, env=env)
    print("Model loaded successfully!")
    
    # Play some episodes
    for episode in range(3):
        print(f"\nStarting episode {episode + 1}")
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Get action from the trained model
            action, _states = model.predict(obs, deterministic=True)
            
            # Take the action
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if steps % 100 == 0:
                print(f"  Step {steps}, Reward: {reward:.3f}, Total: {total_reward:.3f}")
            
            if done:
                print(f"Episode {episode + 1} finished after {steps} steps")
                print(f"Final reward: {total_reward:.3f}")
                break
    
    env.close()
    print("\nPlayback completed!")

def extract_model_weights(model_path):
    """
    Extract the neural network weights from a trained model.
    
    Args:
        model_path: Path to the saved model (.zip file)
    """
    print(f"Extracting weights from: {model_path}")
    
    # Load the model
    model = PPO.load(model_path)
    
    # Access the policy network
    policy = model.policy
    
    print("Model architecture:")
    print(f"  Policy type: {type(policy).__name__}")
    print(f"  Observation space: {policy.observation_space}")
    print(f"  Action space: {policy.action_space}")
    
    # Get the neural network parameters
    if hasattr(policy, 'mlp_extractor'):
        print("\nMLP Extractor parameters:")
        for name, param in policy.mlp_extractor.named_parameters():
            print(f"  {name}: {param.shape}")
    
    if hasattr(policy, 'action_net'):
        print("\nAction network parameters:")
        for name, param in policy.action_net.named_parameters():
            print(f"  {name}: {param.shape}")
    
    if hasattr(policy, 'value_net'):
        print("\nValue network parameters:")
        for name, param in policy.value_net.named_parameters():
            print(f"  {name}: {param.shape}")
    
    return model

if __name__ == "__main__":
    # Example usage
    model_path = "./ppo_sb3_gfootball.zip"  # Path to your trained model
    
    try:
        # Extract and examine the model weights
        model = extract_model_weights(model_path)
        
        # Play with the model
        load_and_play_with_model(model_path, render=True)
        
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print("Please train a model first using run_sb3_ppo.py")
    except Exception as e:
        print(f"Error: {e}") 