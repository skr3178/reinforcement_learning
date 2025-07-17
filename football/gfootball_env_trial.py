import gfootball.env as football_env
env = football_env.create_environment(
      env_name="academy_empty_goal_close", 
      representation='extracted', # pixels, pixels_gray, extracted_stacked, extracted
      # stacked=False,
      stacked= False,
      logdir='/tmp/football', 
      write_goal_dumps=False, 
      write_full_episode_dumps=False, 
      render=True)

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


# def create_single_football_env(iprocess):
#   """Creates gfootball environment."""
#   env = football_env.create_environment(
#       env_name=FLAGS.level, stacked=('stacked' in FLAGS.state),
#       rewards=FLAGS.reward_experiment,
#       logdir=logger.get_dir(),
#       write_goal_dumps=FLAGS.dump_scores and (iprocess == 0),
#       write_full_episode_dumps=FLAGS.dump_full_episodes and (iprocess == 0),
#       # number of envs that are rendered, can be 0 or 2.
#       render=FLAGS.render and (iprocess == 0),
#       dump_frequency=50 if FLAGS.render and iprocess == 0 else 0)