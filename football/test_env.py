import gfootball.env as football_env

# env = football_env.create_environment(
#     env_name="academy_empty_goal_close",
#     representation="simple115v2",
#     # use_sticky_actions=True,
#     render=True,
# )
# step=0
# env.reset()

# for i in range(1000):  
#     obs, reward, done, info = env.step(env.action_space.sample())
#     print(obs)
#     env.render()

# env.close() 

import gfootball.env as football_env
env = football_env.create_environment(env_name="offside_test", stacked=False, logdir='/tmp/football', 
                                      write_goal_dumps=False, write_full_episode_dumps=False, render=True)
obs = env.reset()
steps = 0
for steps in range(1000):
# while True:
  obs, rew, done, info = env.step(env.action_space.sample())
  steps += 1
  if steps % 100 == 0:
    print("Step %d Reward: %f" % (steps, rew))
  if done:
    obs = env.reset()

print("Steps: %d Reward: %.2f" % (steps, rew))