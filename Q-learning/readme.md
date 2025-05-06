How Q learning works
Our goal is to find a policy (actions in each state) that leads to the highest cumulative reward. 
Q-learning learns the best policy through guided sampling.
The agent records the rewards that it gets from actions that it performs in the environment. 
The Q-values are the expected rewards of the actions in the states.
The agent uses the Q-values to guide which actions it will sample. Q-values Q(s, a) are stored in an array that is indexed by state and action. 
The Q-values guide the exploration, and higher values indicate better actions.

It uses an epsilon greedy behavior policy: mostly the best action is followed, but in a certain fraction a random action is chosen, for exploration.

1. Initialize the Q-table to random values.
2. Select a state s.
3. For all possible actions from s, select the one with the highest Q-value and travel
to this state, which becomes the new s, or, with -greedy, explore.
4. Update the values in the Q-array using the equation.
5. Repeat until the goal is reached; when the goal state is

Excerpts from Aske Plaats_ Deep reinforcement learning
