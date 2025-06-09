In this env, 2 functions are defined. 
 
One is the value-iteration function and other the greedy policy function. 
They are similar in their code however, with certain distinctions.

The value-iteration function loops through creating a Value matrix (based on observation space and action space).


The value matrix is further used downstream for the greedy policy which instead of computing the values from the beginning, 
takes in the value as returned from the previous function and outputs the action that provides the best value. 




 (# create a matrix of the best possible state values)
```
def iterate_value_function(v_inp, gamma, env):
    ret = np.zeros(env.observation_space.n)
    for sid in range(env.observation_space.n):
        temp_v = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for (prob, nxt_state, reward, is_final) in env.P[sid][action]:
                # typical output: [(1.0, 100, -1, False)]
                temp_v[action] += prob * (reward + gamma * v_inp[nxt_state] * (not is_final))
        ret[sid] = max(temp_v)
    return ret
    
```
Value function gives overall state matrix of rewards, greedy policy selects the most optimal action

use the value function from the value function iteration to update the policy and determine the highest value action

```
def build_greedy_policy(v_inp, gamma, env):
    new_policy = np.zeros(env.observation_space.n)
    for state_id in range(env.observation_space.n):
        profits = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for (prob, nxt_state, reward, is_final) in env.P[state_id][action]:
                profits[action] += prob * (reward + gamma * v_inp[nxt_state])
        new_policy[state_id] = np.argmax(profits)
    return new_policy
```

