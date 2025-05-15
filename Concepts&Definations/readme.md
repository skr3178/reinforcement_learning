 Model-Based vs. Model-Free Reinforcement Learning
Model-Based:
The agent learns or uses a model of the environment (i.e., transition probabilities and rewards) to simulate outcomes and plan ahead.
➤ Example: Using a model to simulate future states before choosing an action (like MCTS or Dyna-Q).

Model-Free:
The agent learns directly from experience, without learning a model of the environment.
➤ Example: Q-learning, where the agent updates action values from observed rewards and transitions.

🔶 Policy-Based vs. Value-Based Methods
Policy-Based:
Directly learns a policy (π(a|s)) — a mapping from states to actions — usually using gradient methods.
➤ Example: REINFORCE, PPO.

Value-Based:
Learns a value function (like Q(s, a)) and derives the policy by acting greedily with respect to those values.
➤ Example: Q-learning, DQN.
