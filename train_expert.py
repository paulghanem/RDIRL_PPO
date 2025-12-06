import os
import argparse
from datetime import datetime
import torch

from gail_airl_ppo.env import make_env
from gail_airl_ppo.algo import SAC, PPO, AVG
from gail_airl_ppo.trainer import Trainer


def run(args):
    env = make_env(args.env_id)
    env_test = make_env(args.env_id)

    # Select algorithm based on args
    if args.algo == 'avg':
        algo = AVG(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device("cuda" if args.cuda else "cpu"),
            seed=args.seed,
            gamma=args.gamma,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            alpha=args.alpha
        )
        algo.name = 'avg'
    elif args.algo == 'ppo':
        algo = PPO(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device("cuda" if args.cuda else "cpu"),
            seed=args.seed
        )
        algo.name = 'ppo'
    elif args.algo == 'sac':
        algo = SAC(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device("cuda" if args.cuda else "cpu"),
            seed=args.seed
        )
        algo.name = 'sac'
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, args.algo, f'seed{args.seed}-{time}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--algo', type=str, default='avg', choices=['avg', 'ppo', 'sac'],
                   help='Algorithm to use for training expert')
    p.add_argument('--num_steps', type=int, default=500000)
    p.add_argument('--eval_interval', type=int, default=1000)
    p.add_argument('--env_id', type=str, default='HalfCheetah-v4')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    # AVG-specific hyperparameters
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lr_actor', type=float, default=0.0063)
    p.add_argument('--lr_critic', type=float, default=0.0087)
    p.add_argument('--alpha', type=float, default=0.07, help='Entropy coefficient for AVG')
    args = p.parse_args()
    run(args)
