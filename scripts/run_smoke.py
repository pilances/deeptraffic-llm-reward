"""Run short reproducible episodes with the two non-learning smoke policies."""

import argparse
import numpy as np
from statistics import median
from typing import List

from deeptraffic_llm_reward import DeepTrafficConfig, DeepTrafficEnv
from deeptraffic_llm_reward.policies import RandomPolicy, RuleBasedPolicy


def evaluate(policy_name: str, seeds: List[int], frames: int) -> None:
    config = DeepTrafficConfig(episode_frames=frames)
    episode_speeds = []
    override_rates = []

    for seed in seeds:
        env = DeepTrafficEnv(config)
        observation, _ = env.reset(seed=seed)
        policy = RandomPolicy(seed=seed) if policy_name == "random" else RuleBasedPolicy()
        truncated = False
        info = {}
        while not truncated:
            observation, _, _, truncated, info = env.step(policy.act(observation))
        episode_speeds.append(info["episode_mean_speed_mph"])
        override_rates.append(info["safety_override_rate"])

    print(f"policy={policy_name}")
    print(f"seeds={seeds}")
    print(f"environment_version={info['environment_version']}")
    print(f"specification_source={info['specification_source']}")
    print(f"frames_per_episode={frames}")
    print(f"decision_interval_frames={config.decision_frames}")
    print(f"median_mean_speed_mph={median(episode_speeds):.3f}")
    print(f"iqr_mean_speed_mph={np.subtract(*np.percentile(episode_speeds, [75, 25])):.3f}")
    print(f"median_safety_override_rate={median(override_rates):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("random", "rule"), default="rule")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--frames", type=int, default=10_000)
    args = parser.parse_args()
    if args.seeds < 1 or args.frames < 1:
        parser.error("--seeds and --frames must be positive")
    evaluate(args.policy, list(range(args.seeds)), args.frames)


if __name__ == "__main__":
    main()

