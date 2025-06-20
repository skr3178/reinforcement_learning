# runs with env mdpg

import gym
import numpy as np

# env.observation_space.n=500

# create a matrix of the best possible state values
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
# Value function gives overall state matrix of rewards, greedy policy selects the most optimal action

# env.action_space.n =6
# use the value function from the value function iteration to update the policy and determine the highest value action
def build_greedy_policy(v_inp, gamma, env):
    new_policy = np.zeros(env.observation_space.n)
    for state_id in range(env.observation_space.n):
        profits = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for (prob, nxt_state, reward, is_final) in env.P[state_id][action]:
                profits[action] += prob * (reward + gamma * v_inp[nxt_state])
        new_policy[state_id] = np.argmax(profits)
    return new_policy

env = gym.make('Taxi-v3')
# env = gym.make("CartPole-v1")
gamma = 0.9
cum_reward = 0
n_rounds = 500

for t_rounds in range(n_rounds):
    observation = env.reset()
    if isinstance(observation, tuple):  # For gym>=0.26
        observation = observation[0]
    v = np.zeros(env.observation_space.n)

    for _ in range(100):
        v_old = v.copy()
        v = iterate_value_function(v, gamma, env)
        if np.allclose(v, v_old): # checks if two arrays are approximately equal within a tolerance
            break

    policy = build_greedy_policy(v, gamma, env).astype(np.int32)

    for _ in range(1000):
        action = policy[observation]
        step_result = env.step(action)
        if len(step_result) == 5:
            observation, reward, done, truncated, info = step_result
            done = done or truncated
        else:
            observation, reward, done, info = step_result

        cum_reward += reward
        if done:
            break

    if t_rounds % 50 == 0 and t_rounds > 0:
        print(f"Average reward after {t_rounds + 1} episodes: {cum_reward / (t_rounds + 1):.2f}")

env.close()
