# main.py

"""

python3 main.py --FrozenLake --map 4x4

python3 main.py --FrozenLake --map 8x8

python3 main.py --Taxi_v3

python3 main.py --CliffWalking \
  --sims 200 --rollout 120 --ep_steps 200 --episodes 2000 --seed 4



"""



from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, Any, List, Tuple, Optional

import gymnasium as gym
import numpy as np

from ramcts_engine import MCTSSolver, MCTSConfig, NGramMiner
from env_adapters import (
    FrozenLakeModel, GymDiscreteModel,
    success_frozenlake, success_taxi, success_cliff
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int):
        self.Q = np.zeros((n_states, n_actions), dtype=float)
        self.alpha = 0.6
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def choose_action(self, s: int) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.Q.shape[1])
        return int(np.argmax(self.Q[s]))

    def update(self, s: int, a: int, r: float, ns: int, done: bool) -> None:
        td_target = r + (0.0 if done else self.gamma * np.max(self.Q[ns]))
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * td_target

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def run_qlearning_experiment(map_name: str = "4x4",
                             max_episodes: int = 10000,
                             success_streak: int = 10) -> Dict[str, Any]:
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    agent = QLearningAgent(n_states, n_actions)

    episode_returns: List[float] = []
    success_rate: List[float] = []
    solved, streak = False, 0
    solve_ep = -1

    for ep in range(1, max_episodes + 1):
        s, _ = env.reset()
        done = False
        total = 0.0

        while not done:
            a = agent.choose_action(s)
            ns, r, terminated, truncated, _ = env.step(a)
            agent.update(s, a, r, ns, terminated or truncated)
            s = ns
            done = terminated or truncated
            total += r

        agent.decay_epsilon()
        episode_returns.append(total)
        window = episode_returns[-50:]
        success_rate.append(float(np.mean([1.0 if x > 0 else 0.0 for x in window])))

        if total > 0.0:
            streak += 1
        else:
            streak = 0
        if not solved and streak >= success_streak:
            solved, solve_ep = True, ep

        if ep % 50 == 0:
            print(f"[QL] {map_name} | ep={ep:5d} | epsilon={agent.epsilon:.3f} | SR@50={success_rate[-1]:.2f}")

        if solved:
            break

    env.close()
    return {
        "episode_returns": episode_returns,
        "success_rate": success_rate,
        "solved": solved,
        "solve_episode": solve_ep
    }

def run_frozenlake_experiments(run_maps: List[str]) -> None:
    cfg = MCTSConfig(max_sims_per_move=150, rollout_max_steps=50, gamma=1.0)
    summary: Dict[str, Dict[str, int]] = {}
    for map_name in run_maps:
        print("=" * 60)
        print(f"FrozenLake — {map_name}")
        print("=" * 60)

        model = FrozenLakeModel(map_name=map_name)
        env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=False)

        solver = MCTSSolver(model=model, action_count=model.action_count, config=cfg)
        miner = NGramMiner()

        episode_returns: List[float] = []
        success_rate: List[float] = []
        solved, streak = False, 0
        solve_ep = -1

        for ep in range(1, 2000 + 1):
            s, _ = env.reset()
            solver.start_episode(s)
            done = False
            total = 0.0
            trace: List[Tuple[int, int]] = []

            for _ in range(200):
                a = solver.choose_action(history=trace, miner=miner)
                trace.append((s, a))
                s, r, terminated, truncated, _ = env.step(a)
                solver.advance(a, s, edge_reward=r, edge_done=terminated or truncated)
                total += r
                done = terminated or truncated
                if done:
                    break

            miner.update(trace, total)

            episode_returns.append(total)
            window = episode_returns[-50:]
            success_rate.append(float(np.mean([1.0 if x > 0 else 0.0 for x in window])))

            if total > 0.0:
                streak += 1
            else:
                streak = 0

            if not solved and streak >= 10:
                solved, solve_ep = True, ep

            if ep % 50 == 0:
                print(f"[RaMCTS] {map_name} | ep={ep:5d} | SR@50={success_rate[-1]:.2f} | streak={streak}")

            if solved:
                break

        env.close()

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, f"RaMCTS_{map_name}_logs.json"), "w") as f:
            json.dump({"success_rate": success_rate, "episode_returns": episode_returns}, f)

        summary[map_name] = {"RaMCTS": (solve_ep if solved else 2000)}
        print(f"Summary — {map_name}: solved={solved} at ep={solve_ep if solved else 'N/A'}")

    print("Done. Logs saved to:", OUTPUT_DIR)

def run_generic_env(env_id: str,
                    sims_per_move: int,
                    episode_cap: int,
                    rollout_max_steps: int,
                    ep_step_cap: Optional[int],
                    success_kind: str,
                    seed: int = 0) -> None:
    print("=" * 60)
    print(f"RaMCTS — {env_id}")
    print("=" * 60)

    if success_kind == "taxi":
        success_fn = success_taxi
    elif success_kind == "cliff":
        success_fn = success_cliff
    else:
        success_fn = success_frozenlake

    model = GymDiscreteModel(env_id)
    env = model.env
    random.seed(seed)
    np.random.seed(seed)
    try:
        env.reset(seed=seed)
    except Exception:
        pass

    cfg = MCTSConfig(max_sims_per_move=sims_per_move, rollout_max_steps=rollout_max_steps, gamma=1.0)
    solver = MCTSSolver(model=model, action_count=model.action_count, config=cfg)
    miner = NGramMiner()

    solved, streak = False, 0
    last_sr = 0.0
    step_cap = ep_step_cap if ep_step_cap is not None else max(rollout_max_steps, 100)

    for ep in range(1, episode_cap + 1):
        s, _ = env.reset()
        solver.start_episode(s)
        done = False
        total_return = 0.0
        last_reward = 0.0
        last_terminated = False
        trace: List[Tuple[int, int]] = []

        for _ in range(step_cap):  # safety cap per episode
            a = solver.choose_action(history=trace, miner=miner)
            trace.append((s, a))
            s, r, terminated, truncated, _ = env.step(a)
            solver.advance(a, s, edge_reward=r, edge_done=terminated or truncated)
            total_return += r
            last_reward = r
            last_terminated = bool(terminated)
            done = terminated or truncated
            if done:
                break

        miner.update(trace, total_return)

        # For success, use "terminated" rather than generic "done" to avoid time-limit false positives
        if success_fn(last_terminated, last_reward):
            streak += 1
        else:
            streak = 0

        if ep % 50 == 0:
            last_sr = (0.9 * last_sr + 0.1 * (1.0 if streak > 0 else 0.0))
            print(f"[RaMCTS] {env_id} | ep={ep:5d} | streak={streak} | SR~={last_sr:.2f}")

        if streak >= 10:
            print(f"{env_id} solved in {ep} episodes!")
            solved = True
            break

    if not solved:
        print(f"{env_id} not solved within {episode_cap} episodes.")
    try:
        model.close()
    except Exception:
        pass

def parse_args():
    p = argparse.ArgumentParser(description="Run RaMCTS experiments.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--FrozenLake", action="store_true", help="Run FrozenLake experiments (default).")
    g.add_argument("--Taxi_v3", action="store_true", help="Run Taxi-v3 (RaMCTS only).")
    g.add_argument("--CliffWalking", action="store_true", help="Run CliffWalking-v1 (RaMCTS only).")

    p.add_argument("--map", choices=["both", "4x4", "8x8"], default="both", help="FrozenLake map(s) to run.")
    p.add_argument("--sims", type=int, default=None, help="MCTS sims per move (generic envs).")
    p.add_argument("--episodes", type=int, default=None, help="Episode cap (generic envs).")
    p.add_argument("--rollout", type=int, default=None, help="Rollout max steps (generic envs).")
    p.add_argument("--ep_steps", type=int, default=None, help="Max env steps per episode (generic envs).")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.Taxi_v3:
        run_generic_env(
            env_id="Taxi-v3",
            sims_per_move=args.sims or 200,
            episode_cap=args.episodes or 1000,
            rollout_max_steps=args.rollout or 100,
            ep_step_cap=args.ep_steps or 200,
            success_kind="taxi",
            seed=args.seed,
        )
    elif args.CliffWalking:
        run_generic_env(
            env_id="CliffWalking-v1",
            sims_per_move=args.sims or 200,
            episode_cap=args.episodes or 2000,
            rollout_max_steps=args.rollout or 80,
            ep_step_cap=args.ep_steps or 100,
            success_kind="cliff",
            seed=args.seed,
        )
    else:
        if args.map == "both":
            run_maps = ["4x4", "8x8"]
        else:
            run_maps = [args.map]
        run_frozenlake_experiments(run_maps)
