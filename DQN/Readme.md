DQN

Advantages of the DQN over rest.

The original focus of DQN is on breaking correlations between subsequent states
and also on slowing down changes to parameters in the training process to improve stability. The DQN algorithm has two methods to achieve this: (1) experience replay
and (2) infrequent weight updates.We will first look at experience replay.

1. EXPERIENCE REPLAY:
In reinforcement learning training samples are created in a sequence of interactions
with the environment, and subsequent training states are strongly correlated to
preceding states. There is a tendency to train the network on too many samples
of a certain kind or in a certain area, and other parts of the state space remain underexplored.
Furthermore, through function approximation and bootstrapping, some
behavior may be forgotten. When an agent reaches a new level in a game that is
different from previous levels, the agent may forget how to play the other level.
We can reduce correlation—and the local minima they cause—by adding a small
amount of supervised learning. To break correlations and to create a more diverse set
of training examples, DQN uses experience replay. Experience replay introduces a
replay buffer, a cache of previously explored states, from which it samples
training states at random.5 Experience replay stores the last N examples in the
replay memory and samples uniformly when performing updates. A typical number
for N is 106. By using a buffer, a dynamic dataset from which recent training
examples are sampled, we train states from a more diverse set, instead of only from
the most recent one. The goal of experience replay is to increase the independence
of subsequent training examples. The next state to be trained on is no longer a direct
successor of the current state, but one somewhere in a long history of previous states.
In this way the replay buffer spreads out the learning over more previously seen
states, breaking temporal correlations between samples. DQN’s replay buffer (1)
improves coverage and (2) reduces correlation.
DQN treats all examples equal, old, and recent alike. A form of importance
sampling might differentiate between important transitions, as we will see in the
next section.
Note that, curiously, training by experience replay is a form of off-policy
learning, since the target parameters are different from those used to generate the
sample. Off-policy learning is one of the three elements of the deadly triad, and we
find that stable learning can actually be improved by a special form of one of its
problems.

2. INFREQUENT UPDATES OF THE TARGET WEIGHTS
The second improvement in DQN is infrequent weight updates, introduced in the
2015 paper on DQN. The aim of this improvement is to reduce divergence that
is caused by frequent updates of weights of the target Q-value. Again, the aim is to
improve the stability of the network optimization by improving the stability of the
Q-target in the loss function.
Every n updates, the network Q is cloned to obtain target network ˆ Q, which
is used for generating the targets for the following n updates to Q. In the original
DQN implementation, a single set of network weights θ are used, and the network
is trained on a moving loss target. Now, with infrequent updates the weights of the
target network change much slower than those of the behavior policy, improving the
stability of the Q-targets.
The second network improves the stability of Q-learning, where normally an
update to Qθ (st, at ) also changes the target at each time step, quite possibly leading to oscillations and divergence of the policy. Generating the targets using an older set of parameters adds a delay between the time an update to Qθ is made and the time the update changes the targets, making oscillations less likely.

Excerpts from Aske PLaat-Deep reinforcement learning
