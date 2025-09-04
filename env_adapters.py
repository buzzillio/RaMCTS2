# env_adapters.py
from __future__ import annotations
from typing import Tuple, Any, Optional
import random
import gymnasium as gym
import numpy as np

class FrozenLakeModel:
    """Deterministic FrozenLake model (no slip) used for exact rollouts."""
    def __init__(self, map_name: str = "4x4"):
        if map_name == "4x4":
            self.desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
            self.n = 4
        elif map_name == "8x8":
            self.desc = [
                "SFFFFFFF","FFFFFFFF","FFFHFFFF",
                "FFFFFHFF","FFFHFFFF","FHHFFFHF",
                "FHFFHFHF","FFFHFFFG",
            ]
            self.n = 8
        else:
            raise ValueError(f"Unsupported FrozenLake map: {map_name}")
        self.holes = set()
        self.goal = -1
        for r, row in enumerate(self.desc):
            for c, ch in enumerate(row):
                idx = r * self.n + c
                if ch == "H":
                    self.holes.add(idx)
                elif ch == "G":
                    self.goal = idx
        if self.goal < 0:
            raise RuntimeError("Goal cell 'G' not found in the map.")
        self.action_count = 4

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        r, c = divmod(int(state), self.n)
        if action == 0:   # left
            c = max(0, c - 1)
        elif action == 1: # down
            r = min(self.n - 1, r + 1)
        elif action == 2: # right
            c = min(self.n - 1, c + 1)
        elif action == 3: # up
            r = max(0, r - 1)
        ns = r * self.n + c
        if ns in self.holes:
            return ns, 0.0, True
        if ns == self.goal:
            return ns, 1.0, True
        return ns, 0.0, False

class GymDiscreteModel:
    """Generic adapter for Gymnasium toy_text environments with Discrete states/actions."""
    def __init__(self, env_id: str, **make_kwargs):
        self.env_id = env_id
        self.env = gym.make(env_id, **make_kwargs)
        self.action_count = int(self.env.action_space.n)
        base = self.env.unwrapped
        self._use_P = hasattr(base, "P")
        if self._use_P:
            self.P = base.P  # type: ignore
            self._scratch = None
        else:
            scratch = gym.make(env_id, **make_kwargs)
            try:
                scratch.reset()
            except Exception:
                pass
            self._scratch = scratch.unwrapped
            if not hasattr(self._scratch, "s"):
                raise ValueError(f"{env_id} lacks P and unwrapped.s; cannot build generative model.")

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
        if getattr(self, "_scratch", None) is not None:
            try:
                self._scratch.close()  # type: ignore
            except Exception:
                pass

    def _step_via_P(self, state: int, action: int) -> Tuple[int, float, bool]:
        transitions = self.P[state][action]  # type: ignore[index]
        u, cdf = random.random(), 0.0
        for prob, ns, rew, term in transitions:
            cdf += prob
            if u <= cdf:
                return int(ns), float(rew), bool(term)
        prob, ns, rew, term = transitions[-1]
        return int(ns), float(rew), bool(term)

    def _step_via_unwrapped(self, state: int, action: int) -> Tuple[int, float, bool]:
        ue = self._scratch  # type: ignore
        ue.s = int(state)   # type: ignore
        obs, rew, terminated, truncated, _ = ue.step(int(action))  # type: ignore
        if hasattr(ue, "s"):
            ns = int(ue.s)  # type: ignore
        elif isinstance(obs, (int, np.integer)):
            ns = int(obs)
        else:
            raise RuntimeError(f"{self.env_id}: cannot infer next_state from step().")
        done = bool(terminated)
        return ns, float(rew), done

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        if self._use_P:
            return self._step_via_P(state, action)
        else:
            return self._step_via_unwrapped(state, action)

def success_frozenlake(terminated: bool, reward: float) -> bool:
    return terminated and (reward > 0)

def success_taxi(terminated: bool, reward: float) -> bool:
    return terminated and (reward >= 20)

def success_cliff(terminated: bool, reward: float) -> bool:
    # -1 on step (including final to goal), -100 on cliff, terminal at goal.
    return terminated and (reward != -100)
