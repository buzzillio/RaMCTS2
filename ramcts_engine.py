# ramcts_engine.py (improved)
from __future__ import annotations
import math
import random
import collections
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

class Node:
    """Tree node with edge reward bookkeeping."""
    def __init__(self, state: Any, action_count: int,
                 parent: Optional["Node"] = None,
                 action_taken: Optional[int] = None,
                 edge_reward: float = 0.0,
                 edge_done: bool = False):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.edge_reward = float(edge_reward)
        self.edge_done = bool(edge_done)
        self.N = 0
        self.W = 0.0
        self.children: Dict[int, Node] = {}
        self.untried_actions = list(range(action_count))
        random.shuffle(self.untried_actions)

    def q_value(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

@dataclass
class MinerConfig:
    decay: float = 0.999
    prune_threshold: float = 1e-4
    max_table_size: int = 10000
    near_success_quantile: float = 0.5

class NGramMiner:
    def __init__(self, config: Optional[MinerConfig] = None):
        self.config = config or MinerConfig()
        self.c_all = collections.defaultdict(float)
        self.c_pos = collections.defaultdict(float)
        self.episode_returns = collections.deque(maxlen=50)
        self.all_total = 0.0
        self.pos_total = 0.0
        self.pos_updates = 0

    def _apply_decay(self):
        d = self.config.decay
        self.all_total *= d
        self.pos_total *= d
        if len(self.c_all) > self.config.max_table_size:
            for k in list(self.c_all.keys()):
                self.c_all[k] *= d
            for k in list(self.c_pos.keys()):
                self.c_pos[k] *= d
            tiny = [k for k, v in self.c_all.items() if v < self.config.prune_threshold]
            for k in tiny:
                self.c_all.pop(k, None)
                self.c_pos.pop(k, None)

    def _get_near_success_threshold(self) -> float:
        if not self.episode_returns:
            return float("-inf")
        arr = sorted(self.episode_returns)
        idx = max(0, int(self.config.near_success_quantile * (len(arr) - 1)))
        return arr[idx]

    def _extract_ngrams(self, trace: List[Tuple[int, int]]) -> List[tuple]:
        grams = []
        for i in range(len(trace)):
            grams.append((trace[i],))
            if i >= 1:
                grams.append((trace[i-1], trace[i]))
        return grams

    def update(self, trace: List[Tuple[int, int]], total_return: float) -> None:
        self._apply_decay()
        self.episode_returns.append(float(total_return))
        near_thr = self._get_near_success_threshold()
        is_positive = (total_return > near_thr)
        grams = self._extract_ngrams(trace)
        for g in grams:
            self.c_all[g] += 1.0
            if is_positive:
                self.c_pos[g] += 1.0
                self.pos_total += 1.0
        if is_positive:
            self.pos_updates += 1

    def _compute_score(self, gram: tuple) -> float:
        c_all = self.c_all.get(gram, 0.0)
        if c_all == 0:
            return 0.0
        success_rate = self.c_pos.get(gram, 0.0) / (c_all + 2.0)
        freq_bonus = min(1.0, c_all / 20.0)
        return success_rate * (1.0 + freq_bonus)

    def calculate_prior(self, node: Node, action: int,
                        history: List[Tuple[int, int]]) -> float:
        state = node.state
        score = 0.0
        gram1 = ((state, action),)
        score += self._compute_score(gram1)
        if history:
            gram2 = (history[-1], (state, action))
            score += 1.5 * self._compute_score(gram2)
        return max(0.0, float(score))

@dataclass
class MCTSConfig:
    max_sims_per_move: int = 150
    c_uct: float = 1.414
    c_puct: float = 1.0
    beta_max: float = 0.5
    beta_start_pos: int = 2
    beta_full_pos: int = 5
    beta_floor_after_episodes: int = 50
    beta_floor: float = 0.05
    warm_sims: int = 10
    rollout_max_steps: int = 100
    gamma: float = 1.0  # discount for cumulative rewards (1.0 = undiscounted)

class MCTSSolver:
    def __init__(self, model: Any, action_count: int, config: Optional[MCTSConfig] = None):
        self.model = model
        self.config = config or MCTSConfig()
        self.action_count = action_count
        self.root: Optional[Node] = None

    def start_episode(self, start_state: Any):
        self.root = Node(start_state, self.action_count)

    def advance(self, action: int, next_state: Any, edge_reward: float = 0.0, edge_done: bool = False):
        if self.root is None:
            self.root = Node(next_state, self.action_count, edge_reward=edge_reward, edge_done=edge_done)
            return
        child = self.root.children.get(action)
        if child is None:
            child = Node(next_state, self.action_count, parent=None, action_taken=action,
                         edge_reward=edge_reward, edge_done=edge_done)
        self.root = child
        self.root.parent = None

    def _compute_beta(self, miner: Optional[NGramMiner]) -> float:
        if miner is None:
            return 0.0
        if miner.pos_updates < self.config.beta_start_pos and len(miner.episode_returns) >= self.config.beta_floor_after_episodes:
            return min(self.config.beta_floor, self.config.beta_max * 0.1)
        pos = miner.pos_updates
        if pos < self.config.beta_start_pos:
            return 0.0
        if pos >= self.config.beta_full_pos:
            return self.config.beta_max
        progress = (pos - self.config.beta_start_pos) / max(1, (self.config.beta_full_pos - self.config.beta_start_pos))
        return self.config.beta_max * progress

    def _simulate_rollout(self, state: Any) -> float:
        total = 0.0
        gamma = self.config.gamma
        g = 1.0
        for _ in range(self.config.rollout_max_steps):
            action = random.randrange(self.action_count)
            state, reward, done = self.model.step(state, action)
            total += g * float(reward)
            if done:
                return total
            g *= gamma
        return total

    def _select_action_from_node(self, node: Node, miner: Optional[NGramMiner],
                                 beta: float, path_hist: List[Tuple[int, int]],
                                 sim_count: int) -> int:
        if not node.children:
            return random.randrange(self.action_count)
        use_puct = (miner is not None and beta > 0 and sim_count >= self.config.warm_sims and node.N >= 10)
        if use_puct:
            prior_scores = {a: miner.calculate_prior(node, a, path_hist) for a in node.children}
            max_s = max(prior_scores.values())
            if max_s > 0:
                min_s = min(prior_scores.values())
                span = max(1e-9, max_s - min_s)
                priors = {a: (s - min_s) / span for a, s in prior_scores.items()}
            else:
                priors = {a: 0.0 for a in node.children}
            best_action, best_value = None, -float("inf")
            sum_n = max(1, node.N)
            for a, child in node.children.items():
                q = child.q_value()
                u = self.config.c_puct * beta * priors[a] * math.sqrt(sum_n) / (1 + child.N)
                val = q + u
                if val > best_value:
                    best_value, best_action = val, a
            return int(best_action)
        else:
            best_action, best_value = None, -float("inf")
            log_n = math.log(max(1, node.N))
            for a, child in node.children.items():
                if child.N == 0:
                    val = float("inf")
                else:
                    val = child.q_value() + self.config.c_uct * math.sqrt(log_n / child.N)
                if val > best_value:
                    best_value, best_action = val, a
            return int(best_action)

    def _select_and_expand(self, miner: Optional[NGramMiner], beta: float,
                           history: List[Tuple[int, int]], sim_count: int) -> List[Node]:
        assert self.root is not None, "Call start_episode(start_state) before choose_action()."
        path = [self.root]
        node = self.root
        path_hist: List[Tuple[int, int]] = list(history)

        # Selection
        while node.is_fully_expanded() and node.children:
            a = self._select_action_from_node(node, miner, beta, path_hist, sim_count)
            if a not in node.children:
                break
            path_hist.append((node.state, a))
            node = node.children[a]
            path.append(node)

        # Expansion
        if not node.is_fully_expanded():
            idx = random.randrange(len(node.untried_actions))
            a = node.untried_actions.pop(idx)
            next_state, rew, done = self.model.step(node.state, a)
            child = Node(next_state, self.action_count, parent=node, action_taken=a,
                         edge_reward=rew, edge_done=done)
            node.children[a] = child
            path.append(child)

        return path

    @staticmethod
    def _backup(path: List[Node], value: float):
        for n in reversed(path):
            n.N += 1
            n.W += value

    @staticmethod
    def _path_edge_return(path: List[Node], gamma: float = 1.0) -> float:
        # Root has no incoming edge. Sum discounted edge rewards from root->...leaf.
        total = 0.0
        g = 1.0
        for n in path[1:]:
            total += g * getattr(n, 'edge_reward', 0.0)
            g *= gamma
        return total

    def choose_action(self, history: List[Tuple[int, int]],
                      miner: Optional[NGramMiner] = None) -> int:
        assert self.root is not None, "Call start_episode(start_state) before choose_action()."

        for sim in range(self.config.max_sims_per_move):
            beta = self._compute_beta(miner)
            path = self._select_and_expand(miner, beta, history, sim)
            gamma = self.config.gamma
            edge_total = self._path_edge_return(path, gamma=gamma)
            leaf = path[-1]
            if getattr(leaf, 'edge_done', False):
                total_value = edge_total
            else:
                rollout_value = self._simulate_rollout(leaf.state)
                total_value = edge_total + rollout_value
            self._backup(path, total_value)

        if not self.root.children:
            return random.randrange(self.action_count)
        return max(self.root.children, key=lambda a: self.root.children[a].N)

# Backwards compatibility alias
RaCTSMiner = NGramMiner
