How Q learning works
Our goal is to find a policy (actions in each state) that leads to the highest cumulative reward. 
Q-learning learns the best policy through guided sampling.
The agent records the rewards that it gets from actions that it performs in the environment. 
The Q-values are the expected rewards of the actions in the states.
The agent uses the Q-values to guide which actions it will sample. Q-values Q(s, a) are stored in an array that is indexed by state and action. 
The Q-values guide the exploration, and higher values indicate better actions.

It uses an epsilon greedy behavior policy: mostly the best action is followed, but in a certain fraction a random action is chosen, for exploration.
