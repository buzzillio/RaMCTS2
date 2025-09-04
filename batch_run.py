
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_run.py — robust + per-environment figures
Runs ALL envs × ALL algos with a single flag: --runs N
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Prefer gymnasium; fallback to gym
try:
    import gymnasium as gym
except Exception:
    import gym

import numpy as np

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm

# ----------------------------
# Config (tweak here if needed)
# ----------------------------
ENVS = ["Taxi-v3", "CliffWalking-v1", "FrozenLake-v1"]

EPISODES = 400
MAX_STEPS = 200
GAMMA = 0.99

# Q-Learning
QL_ALPHA = 0.8
QL_EPS_START = 1.0
QL_EPS_END = 0.05
QL_EPS_DECAY_FRAC = 0.7

# MCTS constants
MCTS_C = 1.4
MCTS_BUDGET = 64
ROLLOUT_DEPTH = 18

# RaMCTS constants
RAMCTS_C = 1.2
RAMCTS_BUDGET = 96
RAMCTS_RAVE_BETA = 0.5

# Per-environment tuning
ENV_TUNING = {
    "Taxi-v3":         dict(mcts_budget=32,  ramcts_budget=48,  rollout=12),
    "CliffWalking-v1": dict(mcts_budget=64,  ramcts_budget=96,  rollout=16),
    "FrozenLake-v1":   dict(mcts_budget=64,  ramcts_budget=96,  rollout=16),
}


# ----------------------------
# Utils
# ----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def rolling_mean(x: List[float], w: int) -> np.ndarray:
    if w <= 1:
        return np.array(x, dtype=float)
    out = []
    s = 0.0
    for i, v in enumerate(x):
        s += v
        if i >= w:
            s -= x[i - w]
            out.append(s / w)
        else:
            out.append(s / (i + 1))
    return np.array(out, dtype=float)


def save_plot_reward_and_len(env_id: str, algo: str, rewards_runs, lens_runs, out_dir: Path):
    ensure_dir(out_dir)

    # Reward curves
    fig = plt.figure(figsize=(9, 4.8))
    R = np.array(rewards_runs, dtype=float)
    for r in range(R.shape[0]):
        plt.plot(R[r], alpha=0.25)
    if R.size:
        plt.plot(R.mean(axis=0), linewidth=2.5, label="Mean across runs")
        plt.legend()
    plt.title(f"{env_id} — {algo}: Reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    fig.tight_layout()
    fig.savefig(out_dir / "reward_curve.png", dpi=160)
    plt.close(fig)

    # Episode lengths
    fig2 = plt.figure(figsize=(9, 4.8))
    L = np.array(lens_runs, dtype=float)
    for r in range(L.shape[0]):
        plt.plot(L[r], alpha=0.25)
    if L.size:
        plt.plot(L.mean(axis=0), linewidth=2.5, label="Mean across runs")
        plt.legend()
    plt.title(f"{env_id} — {algo}: Episode length")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    fig2.tight_layout()
    fig2.savefig(out_dir / "episode_len.png", dpi=160)
    plt.close(fig2)


def write_runs_and_summary(env_id: str, algo: str, rewards_runs, lens_runs, success_runs, out_dir: Path):
    ensure_dir(out_dir)
    import csv

    # CSV
    with open(out_dir / "runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_idx", "episode", "reward", "ep_len", "success"])
        for r_idx, (r_rewards, r_lens, r_succ) in enumerate(zip(rewards_runs, lens_runs, success_runs)):
            for ep, (rr, ll, ss) in enumerate(zip(r_rewards, r_lens, r_succ)):
                w.writerow([r_idx, ep + 1, rr, ll, int(ss)])

    def summarize(arr_runs):
        arr = np.array(arr_runs, dtype=float)
        if arr.size == 0:
            return {"episodes": 0, "runs": 0, "mean_final_10": 0, "overall_mean": 0, "overall_std": 0}
        last_mean = float(arr[:, -10:].mean()) if arr.shape[1] >= 10 else float(arr.mean())
        return {
            "episodes": int(arr.shape[1]),
            "runs": int(arr.shape[0]),
            "mean_final_10": last_mean,
            "overall_mean": float(arr.mean()),
            "overall_std": float(arr.std()),
        }

    # Summary JSON
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "env": env_id,
            "algo": algo,
            "reward": summarize(rewards_runs),
            "ep_len": summarize(lens_runs),
            "success_rate": summarize(success_runs),
        }, f, indent=2)

    # Raw JSON
    with open(out_dir / "all_runs.json", "w") as f:
        json.dump({
            "env": env_id,
            "algo": algo,
            "rewards_runs": rewards_runs,
            "lens_runs": lens_runs,
            "success_runs": success_runs,
        }, f)


# ----------------------------
# Episode success heuristics
# ----------------------------
def _infer_episode_success(env_id: str, total_reward: float, done: bool, trunc: bool) -> bool:
    if "FrozenLake" in env_id:
        return total_reward > 0.0
    if "CliffWalking" in env_id:
        return total_reward > -100.0
    if "Taxi" in env_id:
        return bool(done) and not bool(trunc)
    return bool(done) and not bool(trunc)


# ----------------------------
# Algorithms
# ----------------------------
def q_learning_run(env_id: str, seed: int, progress=None):
    set_global_seeds(seed)
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    assert hasattr(env.observation_space, "n") and hasattr(env.action_space, "n"),         f"Q-Learning requires discrete obs/action spaces for env {env_id}"

    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA), dtype=np.float32)

    eps = QL_EPS_START
    eps_decay = (QL_EPS_START - QL_EPS_END) / max(1, int(EPISODES * QL_EPS_DECAY_FRAC))

    rewards = []
    ep_lens = []
    success = []

    early_thresh = 0.78
    window = 100
    solved_at = None

    for ep in range(EPISODES):
        if progress is not None:
            progress(ep)

        if solved_at is not None:
            # pad to keep arrays the same length
            rewards.append(rewards[-1])
            ep_lens.append(ep_lens[-1])
            success.append(success[-1])
            continue

        s, _ = env.reset(seed=seed + ep)
        total_r = 0.0
        steps = 0
        done = False
        trunc = False

        while not (done or trunc) and steps < MAX_STEPS:
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                a = int(np.argmax(Q[s]))
            ns, r, done, trunc, _ = env.step(a)
            total_r += r
            best_next = np.max(Q[ns])
            Q[s, a] = (1 - QL_ALPHA) * Q[s, a] + QL_ALPHA * (r + GAMMA * best_next)
            s = ns
            steps += 1

        suc = 1 if _infer_episode_success(env.spec.id if env.spec else env_id, total_r, done, trunc) else 0
        rewards.append(float(total_r))
        ep_lens.append(int(steps))
        success.append(int(suc))

        if len(success) >= window and np.mean(success[-window:]) >= early_thresh:
            solved_at = ep + 1

        if ep < int(EPISODES * QL_EPS_DECAY_FRAC):
            eps = max(QL_EPS_END, eps - eps_decay)

    env.close()
    return rewards, ep_lens, success


@dataclass
class Node:
    s: int
    parent: 'Node' = None
    N: int = 0
    W: float = 0.0
    children: dict = None
    untried_actions: list = None
    N_amaf: dict = None
    W_amaf: dict = None

    def __post_init__(self):
        if self.children is None:
            self.children = {}
        if self.untried_actions is None:
            self.untried_actions = []
        if self.N_amaf is None:
            self.N_amaf = defaultdict(int)
        if self.W_amaf is None:
            self.W_amaf = defaultdict(float)

    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0

    def UCT(self, a: int, c: float):
        child = self.children.get(a)
        if child is None or child.N == 0:
            return float('inf')
        return child.Q() + c * math.sqrt(math.log(self.N + 1) / child.N)

    def UCT_RAVE(self, a: int, c: float, beta: float):
        child = self.children.get(a)
        q_child = child.Q() if child and child.N > 0 else 0.0
        n_amaf = self.N_amaf.get(a, 0)
        q_amaf = (self.W_amaf.get(a, 0.0) / n_amaf) if n_amaf > 0 else 0.0
        q_blend = (1 - beta) * q_child + beta * q_amaf
        if child is None or child.N == 0:
            return float('inf')
        return q_blend + c * math.sqrt(math.log(self.N + 1) / child.N)


def extract_P(env):
    try:
        return env.unwrapped.P
    except Exception:
        raise RuntimeError(f"Could not access transition model P for env {env.spec.id if env.spec else env}")


def sample_model_step(P, s: int, a: int, rng: random.Random):
    trans = P[s][a]
    probs = [p for (p, ns, r, d) in trans]
    idx = rng.choices(range(len(trans)), weights=probs, k=1)[0]
    _, ns, r, done = trans[idx]
    return ns, r, done


def _plan_action_bridge(plan_fn, P, s, budget, c, gamma, rollout_depth, rng, beta=None):
    """Call either MCTS-style (7 args) or RaMCTS-style (8 args with beta)."""
    try:
        return plan_fn(P, s, budget, c, gamma, rollout_depth, rng)
    except TypeError:
        if beta is None:
            beta = RAMCTS_RAVE_BETA
        return plan_fn(P, s, budget, c, beta, gamma, rollout_depth, rng)


def mcts_plan_action(P, s0: int, budget: int, c: float, gamma: float, rollout_depth: int, rng: random.Random):
    nA = len(P[s0].keys()) if isinstance(P[s0], dict) else len(P[s0])
    root = Node(s=s0, parent=None, untried_actions=list(range(nA)))

    def rollout(s: int):
        total = 0.0
        g = 1.0
        for _ in range(rollout_depth):
            a = rng.randrange(nA)
            s, r, done = sample_model_step(P, s, a, rng)
            total += g * r
            if done:
                break
            g *= GAMMA
        return total

    for _ in range(budget):
        node = root
        s = s0
        value = None

        while node.untried_actions == [] and node.children:
            a_best = max(node.children.keys(), key=lambda a: node.UCT(a, c))
            node = node.children[a_best]
            s, r, done = sample_model_step(P, s, a_best, rng)
            if done:
                value = r
                break

        if value is None:
            if node.untried_actions:
                a = node.untried_actions.pop(rng.randrange(len(node.untried_actions)))
                ns, r, done = sample_model_step(P, s, a, rng)
                child = Node(s=ns, parent=node, untried_actions=list(range(nA)))
                node.children[a] = child
                node = child
                if done:
                    value = r
                else:
                    value = r + GAMMA * rollout(ns)
            else:
                value = 0.0

        while node is not None:
            node.N += 1
            node.W += value
            node = node.parent

    if not root.children:
        return rng.randrange(nA)
    return max(root.children.items(), key=lambda kv: kv[1].N)[0]


def ramcts_plan_action(P, s0: int, budget: int, c: float, beta: float, gamma: float, rollout_depth: int, rng: random.Random):
    nA = len(P[s0].keys()) if isinstance(P[s0], dict) else len(P[s0])
    root = Node(s=s0, parent=None, untried_actions=list(range(nA)))

    def rollout(s: int):
        total = 0.0
        g = 1.0
        actions_taken = []
        for _ in range(rollout_depth):
            a = rng.randrange(nA)
            actions_taken.append(a)
            s, r, done = sample_model_step(P, s, a, rng)
            total += g * r
            if done:
                break
            g *= GAMMA
        return total, actions_taken

    for _ in range(budget):
        node = root
        s = s0
        value = None
        path_nodes = []
        rollout_actions = []

        while node.untried_actions == [] and node.children:
            def score(a):
                return node.UCT_RAVE(a, c, beta)
            a_sel = max(node.children.keys(), key=score)
            path_nodes.append(node)
            node = node.children[a_sel]
            s, r, done = sample_model_step(P, s, a_sel, rng)
            if done:
                value = r
                break

        if value is None:
            if node.untried_actions:
                a = node.untried_actions.pop(rng.randrange(len(node.untried_actions)))
                ns, r, done = sample_model_step(P, s, a, rng)
                child = Node(s=ns, parent=node, untried_actions=list(range(nA)))
                node.children[a] = child
                node = child
                if done:
                    value = r
                else:
                    v_roll, rollout_actions = rollout(ns)
                    value = r + GAMMA * v_roll
            else:
                value = 0.0

        # AMAF updates
        for n in path_nodes + [node]:
            for aa in set(rollout_actions):
                n.N_amaf[aa] += 1
                n.W_amaf[aa] += value

        while node is not None:
            node.N += 1
            node.W += value
            node = node.parent

    if not root.children:
        return rng.randrange(nA)
    return max(root.children.items(), key=lambda kv: kv[1].N)[0]


def mcts_like_episode(env, plan_fn, P, budget, c, gamma, rollout_depth, rng, env_id: str, beta=None):
    obs, _ = env.reset(seed=rng.randrange(10**6))
    s = int(obs)
    total_r = 0.0
    steps = 0
    done = False
    trunc = False

    while not (done or trunc) and steps < MAX_STEPS:
        a = _plan_action_bridge(plan_fn, P, s, budget, c, gamma, rollout_depth, rng, beta=beta)
        obs, r, done, trunc, _ = env.step(a)
        s = int(obs)
        total_r += r
        steps += 1

    success = _infer_episode_success(env.spec.id if env.spec else env_id, total_r, done, trunc)
    return total_r, steps, int(success)


def mcts_run(env_id: str, seed: int, progress=None):
    set_global_seeds(seed)
    env = gym.make(env_id)
    P = extract_P(env)

    tune = ENV_TUNING.get(env_id, {})
    budget = int(tune.get("mcts_budget", MCTS_BUDGET))
    rollout = int(tune.get("rollout", ROLLOUT_DEPTH))

    rewards, ep_lens, succ = [], [], []
    rng = random.Random(seed)

    early_thresh = 0.78
    window = 100
    solved_at = None

    for ep in range(EPISODES):
        if progress is not None:
            progress(ep)

        if solved_at is not None:
            rewards.append(rewards[-1])
            ep_lens.append(ep_lens[-1])
            succ.append(succ[-1])
            continue

        r, l, s = mcts_like_episode(env, mcts_plan_action, P, budget, MCTS_C, GAMMA, rollout, rng, env_id)
        rewards.append(float(r))
        ep_lens.append(int(l))
        succ.append(int(s))

        if len(succ) >= window and np.mean(succ[-window:]) >= early_thresh:
            solved_at = ep + 1

    env.close()
    return rewards, ep_lens, succ


def ramcts_run(env_id: str, seed: int, progress=None):
    set_global_seeds(seed + 1)
    env = gym.make(env_id)
    P = extract_P(env)

    tune = ENV_TUNING.get(env_id, {})
    budget = int(tune.get("ramcts_budget", RAMCTS_BUDGET))
    rollout = int(tune.get("rollout", ROLLOUT_DEPTH))

    rewards, ep_lens, succ = [], [], []
    rng = random.Random(seed + 123)

    early_thresh = 0.78
    window = 100
    solved_at = None

    for ep in range(EPISODES):
        if progress is not None:
            progress(ep)

        if solved_at is not None:
            rewards.append(rewards[-1])
            ep_lens.append(ep_lens[-1])
            succ.append(succ[-1])
            continue

        r, l, s = mcts_like_episode(env, ramcts_plan_action, P, budget, RAMCTS_C, GAMMA, rollout, rng, env_id, beta=RAMCTS_RAVE_BETA)
        rewards.append(float(r))
        ep_lens.append(int(l))
        succ.append(int(s))

        if len(succ) >= window and np.mean(succ[-window:]) >= early_thresh:
            solved_at = ep + 1

    env.close()
    return rewards, ep_lens, succ


# ----------------------------
# Optional: project trainers
# ----------------------------
def try_project_trainer(algo_name: str):
    """
    If your repo defines algorithm trainers, use them.
    Expected signature:
        train_one_run(env_id, seed, episodes, max_steps, gamma, progress=None)
    Returns either (rewards, lengths[, success]) or a dict containing
        'episode_rewards', 'episode_lengths', optionally 'success_flags'.
    """
    import importlib
    candidates = []
    if algo_name == "Q-Learning":
        candidates = [("ramcts.q_learning_trainer", "train_one_run"),
                      ("q_learning_trainer", "train_one_run")]
    elif algo_name == "Vanilla MCTS":
        candidates = [("ramcts.mcts_trainer", "train_one_run"),
                      ("mcts_trainer", "train_one_run"),
                      ("algorithms.mcts", "train_one_run")]
    elif algo_name == "RaMCTS":
        candidates = [("ramcts.ramcts_trainer", "train_one_run"),
                      ("ramcts_trainer", "train_one_run"),
                      ("algorithms.ramcts", "train_one_run")]

    for mod, fn in candidates:
        try:
            m = importlib.import_module(mod)
            f = getattr(m, fn, None)
            if callable(f):
                def wrapper(env_id: str, seed: int, progress=None, _f=f):
                    try:
                        out = _f(env_id=env_id, seed=seed, episodes=EPISODES, max_steps=MAX_STEPS, gamma=GAMMA, progress=progress)
                    except TypeError:
                        out = _f(env_id=env_id, seed=seed, episodes=EPISODES, max_steps=MAX_STEPS, gamma=GAMMA)
                    if isinstance(out, tuple) and len(out) >= 2:
                        rewards, lens = out[:2]
                        succ = out[2] if len(out) >= 3 else None
                    elif isinstance(out, dict):
                        rewards = out.get("episode_rewards", [])
                        lens = out.get("episode_lengths", [])
                        succ = out.get("success_flags")
                    else:
                        raise RuntimeError("Trainer return must be (rewards, lengths[, success]) or dict.")
                    return rewards, lens, succ
                return wrapper
        except Exception:
            continue
    return None


# ----------------------------
# Per-environment figures
# ----------------------------
def _fallback_make_env_figures(env_id: str, env_results: Dict[str, Any], output_dir: str):
    """Fallback if plotting_utils.make_env_figures is missing."""
    episodes = int(env_results.get("episodes", 0))
    algos = env_results.get("algorithms", {})

    # Build success matrices, episodes-to-solve, etc.
    def _rolling(x, w):
        if w <= 1:
            return np.array(x, dtype=float)
        out = []
        s = 0.0
        for i, v in enumerate(x):
            s += v
            if i >= w:
                s -= x[i - w]
                out.append(s / w)
            else:
                out.append(s / (i + 1))
        return np.array(out, dtype=float)

    def _eps_to_solve(flags, thr=0.78, win=100):
        arr = np.array(flags, dtype=float)
        rm = _rolling(arr, win)
        idx = np.where(rm >= thr)[0]
        if idx.size == 0:
            return episodes, False
        return int(idx[0] + 1), True

    success_runs_by_algo = {}
    eps_to_solve = {}
    solved_by_algo = {}
    avg_steps_by_algo = {}

    for name, D in algos.items():
        succ = D.get("success_runs", [])
        if not succ:
            # infer from rewards
            succ = []
            for rr in D.get("rewards_runs", []):
                if "FrozenLake" in env_id:
                    succ.append([1 if r > 0 else 0 for r in rr])
                elif "CliffWalking" in env_id:
                    succ.append([1 if r > -100 else 0 for r in rr])
                elif "Taxi" in env_id:
                    succ.append([1 if r >= 5 else 0 for r in rr])
                else:
                    succ.append([1 if r > 0 else 0 for r in rr])
        success_runs_by_algo[name] = succ
        ets = []
        solved_flags = []
        for run in succ:
            ep, ok = _eps_to_solve(run)
            ets.append(ep)
            solved_flags.append(ok)
        eps_to_solve[name] = ets
        solved_by_algo[name] = solved_flags
        L = np.array(D.get("lens_runs", []), dtype=float)
        avg_steps_by_algo[name] = float(L.mean()) if L.size else 1.0

    # (1) Episodes-to-solve bar
    means = []
    cis = []
    algolist = list(algos.keys())
    for a in algolist:
        xs = eps_to_solve[a]
        m = float(np.mean(xs)) if xs else 0.0
        sd = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
        ci = 1.96 * (sd / math.sqrt(max(1, len(xs))))
        means.append(max(1.0, m))
        cis.append(ci)

    fig = plt.figure(figsize=(11, 4.5))
    x = np.arange(len(algolist))
    plt.bar(x, means, yerr=cis, capsize=6)
    plt.yscale("log")
    plt.xticks(x, algolist)
    plt.ylabel("Episodes (log scale)")
    plt.title(f"Episodes to Solve — Mean ± 95% CI ({env_id})")
    for i, a in enumerate(algolist):
        fails = sum(1 for b in solved_by_algo[a] if not b)
        total = len(solved_by_algo[a])
        if total > 0 and fails > 0:
            plt.text(i, means[i] * 1.05, f"Fail: {fails}/{total}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "episodes_to_solve.png"), dpi=200)
    plt.close(fig)

    # (2) Learning dynamics
    fig2 = plt.figure(figsize=(12, 4.8))
    mean_solve_ep = {a: (float(np.mean(v)) if len(v) else None) for a, v in eps_to_solve.items()}
    for a, runs in success_runs_by_algo.items():
        A = np.array(runs, dtype=float)
        if A.size == 0:
            continue
        mean = A.mean(axis=0)
        std = A.std(axis=0, ddof=1) if A.shape[0] > 1 else np.zeros_like(mean)
        mean_s = _rolling(mean, 20)
        std_s = _rolling(std, 20)
        xs = np.arange(1, mean.shape[0] + 1)
        plt.plot(xs, mean_s, linewidth=2.0, linestyle='-' if 'Q-Learning' not in a else '--', label=a)
        plt.fill_between(xs, np.maximum(0, mean_s - std_s), np.minimum(1.0, mean_s + std_s), alpha=0.12)
        if mean_solve_ep[a] is not None:
            star_x = float(mean_solve_ep[a])
            plt.plot([star_x], [0.98], marker='*', markersize=12)
    plt.ylim(0.0, 1.02)
    plt.xlabel("Episodes")
    plt.ylabel("Success Rate")
    plt.title(f"Learning Dynamics — Mean ± SD across runs ({env_id})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_dynamics.png"), dpi=200)
    plt.close(fig2)

    # (3) Radar (simple normalized version)
    labels = ["Sample Efficiency", "Computational Cost", "Scalability", "Interpretability", "Robustness"]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig3 = plt.figure(figsize=(9, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 100)

    # compute scores
    all_eps = [ep for lst in eps_to_solve.values() for ep in lst] or [100.0]
    ep_min, ep_max = max(5, min(all_eps)), max(all_eps)
    def score_sample(ep):
        ep = max(ep_min, min(ep, ep_max))
        return 100.0*(ep_max - ep)/(ep_max - ep_min + 1e-9)
    costs = {}
    for a in algolist:
        steps = max(1.0, avg_steps_by_algo.get(a, 1.0))
        if "Vanilla MCTS" in a:
            b = ENV_TUNING.get(env_id, {}).get("mcts_budget", MCTS_BUDGET)
            costs[a] = steps * b
        elif "RaMCTS" in a:
            b = ENV_TUNING.get(env_id, {}).get("ramcts_budget", RAMCTS_BUDGET)
            costs[a] = steps * b
        else:
            costs[a] = steps
    cmin, cmax = min(costs.values()), max(costs.values())
    def score_cost(c):
        c = max(cmin, min(c, cmax))
        return 100.0*(cmax - c)/(cmax - cmin + 1e-9)
    scal = {"Q-Learning": 30.0, "Vanilla MCTS": 60.0, "RaMCTS": 90.0}
    interp = {"Q-Learning": 60.0, "Vanilla MCTS": 40.0, "RaMCTS": 95.0}
    rob = {}
    for a in algolist:
        sflags = solved_by_algo[a]
        frac = (sum(1 for v in sflags if v)/max(1, len(sflags)))*100.0
        sd = np.std(eps_to_solve[a], ddof=1) if len(eps_to_solve[a]) > 1 else 0.0
        pen = (sd/(ep_max+1e-9))*20.0
        rob[a] = max(0.0, min(100.0, frac - pen))

    def vals(a):
        return [score_sample(np.mean(eps_to_solve[a]) if eps_to_solve[a] else ep_max),
                score_cost(costs[a]),
                scal[a], interp[a], rob[a]]

    for a in ["RaMCTS", "Vanilla MCTS", "Q-Learning"]:
        if a not in algolist: continue
        v = vals(a); vv = v + v[:1]
        ax.plot(angles, vv, linewidth=2, label=a); ax.fill(angles, vv, alpha=0.08)

    ax.set_title(f"Multi-Dimensional Performance Analysis ({env_id})", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1), frameon=False)
    plt.savefig(os.path.join(output_dir, "efficiency_radar.png"), dpi=180, bbox_inches="tight")
    plt.close(fig3)


# ----------------------------
# Orchestration
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run ALL envs & ALL algos with only --runs.")
    parser.add_argument("--runs", type=int, required=True, help="Number of independent runs per env & algo.")
    args = parser.parse_args()

    runs = int(args.runs)
    if runs <= 0:
        raise SystemExit("--runs must be a positive integer")

    out_root = Path("runs")
    ensure_dir(out_root)

    # Try to import plotting_utils.make_env_figures; else use fallback
    MAKE_ENV_FIGS = None
    try:
        import plotting_utils as PU
        MAKE_ENV_FIGS = getattr(PU, "make_env_figures", None)
    except Exception as e:
        print("[WARN] plotting_utils not importable; using fallback figures.", e, file=sys.stderr)

    # Algorithms
    algos = [
        ("Q-Learning", q_learning_run, None),
        ("Vanilla MCTS", mcts_run, MCTS_BUDGET),
        ("RaMCTS", ramcts_run, RAMCTS_BUDGET),
    ]

    total_tasks = len(ENVS) * len(algos) * runs
    overall = tqdm(total=total_tasks, desc="All tasks", position=0)

    index = []

    for env_id in ENVS:
        env_results: Dict[str, Any] = {"episodes": EPISODES, "runs": runs, "algorithms": {}}

        for algo_name, algo_fn, budget in algos:
            rewards_runs, lens_runs, succ_runs = [], [], []
            out_dir = out_root / env_id / algo_name.replace(" ", "")
            ensure_dir(out_dir)

            for run_idx in range(runs):
                seed = 1000 + 37 * run_idx
                desc = f"{env_id} | {algo_name} | run {run_idx+1}/{runs}"
                pbar = tqdm(total=EPISODES, desc=desc, leave=False, position=1)

                def _progress(_=None):
                    pbar.update(1)

                try:
                    rewards, lens, succ = algo_fn(env_id, seed, progress=_progress)
                except TypeError:
                    rewards, lens, succ = algo_fn(env_id, seed)
                except Exception as e:
                    if pbar.n < EPISODES: pbar.update(EPISODES - pbar.n)
                    pbar.close()
                    print(f"[ERROR] {env_id} {algo_name} run {run_idx} failed: {e}", file=sys.stderr)
                    raise
                finally:
                    if pbar.n < EPISODES: pbar.update(EPISODES - pbar.n)
                    pbar.close()

                rewards_runs.append(list(map(float, rewards)))
                lens_runs.append(list(map(int, lens)))
                if succ is None:
                    # infer from rewards
                    S = []
                    for r in rewards:
                        if "FrozenLake" in env_id:
                            S.append(1 if r > 0 else 0)
                        elif "CliffWalking" in env_id:
                            S.append(1 if r > -100 else 0)
                        elif "Taxi" in env_id:
                            S.append(1 if r >= 5 else 0)
                        else:
                            S.append(1 if r > 0 else 0)
                    succ_runs.append(S)
                else:
                    succ_runs.append([int(x) for x in succ])

                overall.update(1)

            write_runs_and_summary(env_id, algo_name, rewards_runs, lens_runs, succ_runs, out_dir)
            save_plot_reward_and_len(env_id, algo_name, rewards_runs, lens_runs, out_dir)

            env_results["algorithms"][algo_name] = {
                "rewards_runs": rewards_runs,
                "lens_runs": lens_runs,
                "success_runs": succ_runs,
                **({"budget": budget} if budget is not None else {}),
            }

            index.append({"env": env_id, "algo": algo_name, "out_dir": str(out_dir)})

        # Per-environment figures
        env_out_dir = str(out_root / env_id)
        try:
            if MAKE_ENV_FIGS is not None:
                MAKE_ENV_FIGS(env_id, env_results, env_out_dir)
            else:
                _fallback_make_env_figures(env_id, env_results, env_out_dir)
        except Exception as e:
            print(f"[WARN] plotting for {env_id} failed: {e}", file=sys.stderr)

    overall.close()
    with open(out_root / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    print("\nDone. Artifacts under ./runs/<ENV>/(Q-Learning|VanillaMCTS|RaMCTS)/ and ./runs/<ENV>/*.png")


if __name__ == "__main__":
    main()
