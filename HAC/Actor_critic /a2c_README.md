# A2C (Advantage Actor-Critic) Implementation

This is a simple implementation of the A2C (Advantage Actor-Critic) reinforcement learning algorithm to help understand its core concepts.

## What is A2C?

A2C is an on-policy reinforcement learning algorithm that combines:

1. **Actor (Policy Network)**: Decides which actions to take in a given state
2. **Critic (Value Network)**: Evaluates how good those actions are by estimating the value function

The key advantage of A2C is that it reduces the variance of policy gradient updates by using the advantage function, which is the difference between the action-value function Q(s,a) and the state-value function V(s).

## Key Concepts

- **Actor-Critic Architecture**: Combines policy-based (actor) and value-based (critic) methods
- **Advantage Function**: A(s,a) = Q(s,a) - V(s), used to reduce variance in policy updates
- **On-Policy Learning**: Uses current policy to collect experiences (unlike off-policy methods)
- **Entropy Regularization**: Encourages exploration by adding an entropy bonus to the loss function

## Implementation Details

The implementation consists of:

1. **ActorCritic Network**: A neural network with shared layers and separate heads for the actor and critic
2. **A2C Agent**: Handles the training process, including collecting experiences and updating the network
3. **Training Function**: Runs the algorithm on a specified environment

## How to Use

### Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install gymnasium torch numpy matplotlib
```

### Running the Example

To run the example on the CartPole environment:

```bash
python a2c_example.py
```

This will:
1. Train an A2C agent on the CartPole-v1 environment
2. Plot the learning curve
3. Test the trained agent

### Using the A2C Implementation in Your Own Projects

You can import the A2C implementation in your own projects:

```python
from a2c import A2C, train

# Train an agent on a specific environment
agent, rewards = train(
    env_name="YourEnvironment-v1",
    num_episodes=500,
    max_steps=500,
    gamma=0.99,
    lr=0.001,
    render=False
)

# Use the trained agent
state = env.reset()
action, _ = agent.network.get_action(state)
```

## Customization

You can customize the implementation by:

- Changing the network architecture in the `ActorCritic` class
- Adjusting hyperparameters like learning rate, discount factor, etc.
- Adding additional features like GAE (Generalized Advantage Estimation)

## Differences from A3C

A2C (Advantage Actor-Critic) is the synchronous version of A3C (Asynchronous Advantage Actor-Critic). The main differences are:

- A2C is synchronous, while A3C is asynchronous (uses multiple workers in parallel)
- A2C typically has more stable training due to its synchronous updates
- A3C can be faster in wall-clock time due to parallel data collection

## References

- Mnih, V., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. ICML 2016.
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.