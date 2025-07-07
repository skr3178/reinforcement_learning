import argparse
import os
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gfootball.env as football_env


def parse_args():
    parser = argparse.ArgumentParser(description='Train PPO agent on GFootball with stable-baselines3')
    parser.add_argument('--level', type=str, default='11_vs_11_easy_stochastic', help='Defines type of problem being solved')
    parser.add_argument('--state', type=str, default='extracted_stacked', choices=['extracted', 'extracted_stacked'], help='Observation to be used for training.')
    parser.add_argument('--reward_experiment', type=str, default='scoring,checkpoints', choices=['scoring', 'scoring,checkpoints'], help='Reward to be used for training.')
    parser.add_argument('--policy', type=str, default='CnnPolicy', choices=['CnnPolicy', 'MlpPolicy'], help='Policy architecture')
    parser.add_argument('--num_timesteps', type=int, default=100000, help='Number of timesteps to run for.') #50M typical
    parser.add_argument('--num_envs', type=int, default=16, help='Number of environments to run in parallel.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--lr', type=float, default=0.000343, help='Learning rate')
    parser.add_argument('--ent_coef', type=float, default=0.003, help='Entropy coefficient')
    parser.add_argument('--gamma', type=float, default=0.993, help='Discount factor')
    parser.add_argument('--cliprange', type=float, default=0.08, help='Clip range')
    parser.add_argument('--max_grad_norm', type=float, default=0.64, help='Max gradient norm (clipping)')
    parser.add_argument('--render', type=bool, default=False, help='If True, environment rendering is enabled.')
    parser.add_argument('--dump_full_episodes', type=bool, default=False, help='If True, trace is dumped after every episode.')
    parser.add_argument('--dump_scores', action='store_true', help='If True, sampled traces after scoring are dumped.')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load initial checkpoint from.')
    parser.add_argument('--save_path', type=str, default='./ppo_sb3_gfootball', help='Path to save the model.')
    return parser.parse_args()


def make_env(rank, args):
    def _init():
        # Create log directory if dumping is enabled
        logdir = None
        if args.dump_full_episodes or args.dump_scores:
            logdir = f"./gfootball_dumps/env_{rank}"
            os.makedirs(logdir, exist_ok=True)
        
        env = football_env.create_environment(
            env_name=args.level,
            stacked=('stacked' in args.state),
            rewards=args.reward_experiment,
            logdir=logdir,
            write_goal_dumps=args.dump_scores and (rank == 0),
            write_full_episode_dumps=args.dump_full_episodes and (rank == 0),
            render=args.render and (rank == 0),
            dump_frequency=50 if args.render and rank == 0 else 0
        )
        return env
    return _init


def main():
    args = parse_args()
    ncpu = multiprocessing.cpu_count()
    env_fns = [make_env(i, args) for i in range(args.num_envs)]
    vec_env = SubprocVecEnv(env_fns) if args.num_envs > 1 else DummyVecEnv([make_env(0, args)])

    model = PPO(
        args.policy,
        vec_env,
        verbose=1,
        seed=args.seed,
        n_steps=512,
        batch_size=512,
        n_epochs=2,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        clip_range=args.cliprange,
        max_grad_norm=args.max_grad_norm,
        tensorboard_log="./ppo_sb3_tensorboard/"
    )

    if args.load_path and os.path.exists(args.load_path):
        print(f"Loading model from {args.load_path}")
        model = PPO.load(args.load_path, env=vec_env)

    model.learn(total_timesteps=args.num_timesteps)
    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")

    vec_env.close()

if __name__ == '__main__':
    main()