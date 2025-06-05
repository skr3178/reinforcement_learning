MTCS module calls the Policy_Player_MCTS .py code.

Policy_Player_MTCS calls the explore module

Explore module calls upon the rollout

self.T / self.N is the estimated Q-value — the average reward from this node:

Q(s,a)= N(s,a)/ T(s,a)
 
UCB(s,a) = Q(s,a) + C⋅ sqrt(N(s,a)lnN(s))

T/N= Q-value or the avergae expected reward

