import numpy as np
import gym
# import gymnasium as gym

def temporal_difference(n_samples, alpha, gamma):
    # Initialize Q-values
    Qsa = np.zeros((env.observation_space.n, env.action_space.n))
    s = env.reset()[0]
    # s = int(s)

    for t in range(n_samples):
        a = select_action(s, Qsa)
        s_next, r, done, truncated, info = env.step(a)

        # Update Q function each time step with max of action values
        Qsa[s, a] = Qsa[s, a] + alpha * (r + gamma * np.max(Qsa[s_next]) - Qsa[s, a])

        if done:
            s = env.reset()[0]
        else:
            s = s_next

    return Qsa

def select_action(s, Qsa):
    # Policy is epsilon-greedy
    epsilon = 0.1
    if np.random.rand() < epsilon:
        a = np.random.randint(low=0, high=env.action_space.n)
    else:
        a = np.argmax(Qsa[s])
    return a

# Create environment
env = gym.make('Taxi-v3')

# Run temporal difference learning
Q_values = temporal_difference(n_samples=10000, alpha=0.1, gamma=0.99)

# Close the environment
env.close()