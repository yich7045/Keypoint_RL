import argparse
import os
from datetime import datetime

import torch
import minitouch.env
import gym
from slac.algo import SlacAlgorithm
from slac.env import make_dmc
from slac.trainer import Trainer


def main(args):
    env = make_dmc()

    log_dir = os.path.join(
        "logs",
        f"{args.domain_name}-{args.task_name}",
        f'slac-seed{args.seed}-{datetime.now().strftime("%Y%m%d-%H%M")}',
    )
    algo = SlacAlgorithm()
    trainer = Trainer(
        env=env,
        algo=algo,
        log_dir=log_dir,
        seed=args.seed,
        num_steps=args.num_steps,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=5 * 10 ** 7)
    parser.add_argument("--domain_name", type=str, default="pushing")
    parser.add_argument("--task_name", type=str, default="run")
    parser.add_argument("--action_repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    main(args)
