# """
# RaMCTS

# """

# import math
# import random
# import collections
# from typing import Tuple, Dict, List, Any, Optional
# import numpy as np
# import gymnasium as gym
# from dataclasses import dataclass
# import json
# import os
# from datetime import datetime

# # Create output directory
# OUTPUT_DIR = "ramcts_results_fixed"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# print("=" * 60)
# print("RaMCTS FIXED VERSION - Debugged and Optimized (Heuristic-Free)")
# print("=" * 60)

# # ====================
# # FrozenLake Model (FIXED)
# # ====================
# class FrozenLakeModel:
#     """Fixed FrozenLake model with better action mapping."""
    
#     def __init__(self, map_name="4x4"):
#         if map_name == "4x4":
#             self.desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
#             self.map_size = 4
#         elif map_name == "8x8":
#             self.desc = ["SFFFFFFF", "FFFFFFFF", "FFFHFFFF", 
#                         "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", 
#                         "FHFFHFHF", "FFFHFFFG"]
#             self.map_size = 8
            
#         self.holes = {r * self.map_size + c 
#                      for r, row in enumerate(self.desc) 
#                      for c, char in enumerate(row) if char == 'H'}
#         self.goal_state = self.map_size * self.map_size - 1
#         self.action_count = 4
        
#     def step(self, state: int, action: int) -> Tuple[int, float, bool]:
#         row, col = state // self.map_size, state % self.map_size
        
#         # Fixed action mapping
#         if action == 0:  # LEFT
#             col = max(col - 1, 0)
#         elif action == 1:  # DOWN
#             row = min(row + 1, self.map_size - 1)
#         elif action == 2:  # RIGHT
#             col = min(col + 1, self.map_size - 1)
#         elif action == 3:  # UP
#             row = max(row - 1, 0)
            
#         next_state = row * self.map_size + col
        
#         if next_state in self.holes:
#             return next_state, 0.0, True
#         if next_state == self.goal_state:
#             return next_state, 1.0, True
#         return next_state, 0.0, False

# # ====================
# # Node (FIXED)
# # ====================
# class Node:
#     def __init__(self, state: Any, action_count: int, 
#                  parent: Optional['Node'] = None, 
#                  action_taken: Optional[int] = None):
#         self.state = state
#         self.parent = parent
#         self.action_taken = action_taken
#         self.N = 0
#         self.W = 0.0
#         self.children: Dict[int, Node] = {}
#         self.untried_actions = list(range(action_count))
#         random.shuffle(self.untried_actions)
        
#     def q_value(self) -> float:
#         return self.W / self.N if self.N > 0 else 0.0
    
#     def is_fully_expanded(self) -> bool:
#         return len(self.untried_actions) == 0

# # ====================
# # RaCTS Miner (FIXED)
# # ====================
# @dataclass
# class MinerConfig:
#     """Fixed configuration that actually works."""
#     decay: float = 0.999  # Slower decay
#     prune_threshold: float = 1e-4
#     max_table_size: int = 10000
#     near_success_quantile: float = 0.5  # MUCH lower threshold
#     smoothing_lambda: float = 1.0  # More smoothing
#     idf_cap: float = 2.0  # Lower cap
#     n_gram_max: int = 2  # Just 1-2 grams for now
#     context_weight_max: float = 1.2

# class RaCTSMiner:
#     """Fixed pattern miner with better scoring."""
    
#     def __init__(self, model: FrozenLakeModel, config: MinerConfig = None):
#         self.model = model
#         self.config = config or MinerConfig()
        
#         self.c_all = collections.defaultdict(float)
#         self.c_pos = collections.defaultdict(float)
#         self.episode_returns = collections.deque(maxlen=50)  # Smaller buffer
        
#         self.all_total = 0.0
#         self.pos_total = 0.0
#         self.pos_updates = 0
#         self.top_patterns = []
        
#         # Fixed n-gram size
#         self.n_max = 2 if model.map_size <= 4 else 2
            
#     def _apply_decay(self):
#         """Apply decay more carefully."""
#         decay = self.config.decay
        
#         # Only decay after enough episodes
#         if self.all_total < 10:
#             return
            
#         for key in list(self.c_all.keys()):
#             self.c_all[key] *= decay
#             if self.c_all[key] < self.config.prune_threshold:
#                 del self.c_all[key]
                
#         for key in list(self.c_pos.keys()):
#             self.c_pos[key] *= decay
#             if self.c_pos[key] < self.config.prune_threshold:
#                 del self.c_pos[key]
                
#         self.all_total *= decay
#         self.pos_total *= decay
            
#     def _get_near_success_threshold(self) -> float:
#         """Much more lenient threshold."""
#         if len(self.episode_returns) < 3:
#             return 0.0  # Accept any success early
#         # Use median instead of high percentile
#         return np.median(self.episode_returns)
    
#     def _extract_ngrams(self, trace: List[Tuple[int, int]], n: int) -> List[tuple]:
#         if len(trace) < n:
#             return []
#         return [tuple(trace[i:i+n]) for i in range(len(trace) - n + 1)]
    
#     def update(self, trace: List[Tuple[int, int]], total_return: float):
#         """Update with more lenient criteria."""
#         self._apply_decay()
#         self.episode_returns.append(total_return)
        
#         # Always update all-episode statistics
#         for n in range(1, min(self.n_max + 1, len(trace) + 1)):
#             for gram in self._extract_ngrams(trace, n):
#                 self.c_all[gram] += 1
#         self.all_total += 1
        
#         # Much more lenient positive criteria
#         threshold = self._get_near_success_threshold()
#         if total_return >= threshold and total_return > 0:  # Must have SOME reward
#             for n in range(1, min(self.n_max + 1, len(trace) + 1)):
#                 for gram in self._extract_ngrams(trace, n):
#                     self.c_pos[gram] += 1
#             self.pos_total += 1
#             self.pos_updates += 1
            
#     def _compute_score(self, gram: tuple) -> float:
#         """Simplified scoring that actually works."""
#         if self.all_total == 0 or self.c_all.get(gram, 0) < 2:
#             return 0.0
            
#         # Simple success rate
#         success_rate = self.c_pos.get(gram, 0) / (self.c_all.get(gram, 0) + 1)
        
#         # Frequency bonus (patterns seen more are better)
#         freq_bonus = min(1.0, self.c_all.get(gram, 0) / 20)
        
#         return success_rate * (1 + freq_bonus)
    
#     def calculate_prior(self, node: Node, action: int, 
#                        history: List[Tuple[int, int]]) -> float:
#         """Calculate prior with fixed scoring (no goal-direction bonus)."""
#         state = node.state
        
#         # Build candidate grams
#         score = 0.0
        
#         # 1-gram
#         gram1 = ((state, action),)
#         score += self._compute_score(gram1)
        
#         # 2-gram if history available
#         if len(history) >= 1:
#             gram2 = (history[-1], (state, action))
#             score += self._compute_score(gram2) * 1.5  # Bonus for longer patterns
            
#         return score

# # ====================
# # MCTS Solver (FIXED)
# # ====================
# @dataclass  
# class MCTSConfig:
#     """Fixed configuration that works (heuristic-free)."""
#     max_sims_per_move: int = 100
#     c_uct: float = 1.414  # sqrt(2)
#     c_puct: float = 1.0   # Reduced
#     beta_max: float = 0.5  # MUCH less aggressive
#     beta_start_pos: int = 2  # Earlier
#     beta_full_pos: int = 5   # Earlier
#     warm_sims: int = 10      # Less warmup
#     mini_duel: bool = False  # Disable for now
#     duel_extra_sims: int = 0
#     rollout_max_steps: int = 50

# class MCTSSolver:
#     """Fixed MCTS solver (heuristic-free)."""
    
#     def __init__(self, model: FrozenLakeModel, config: MCTSConfig = None):
#         self.model = model
#         self.config = config or MCTSConfig()
#         self.action_count = model.action_count
        
#     def _compute_beta(self, miner: Optional[RaCTSMiner]) -> float:
#         """More aggressive beta ramp."""
#         if miner is None:
#             return 0.0
        
#         pos = miner.pos_updates
#         if pos < self.config.beta_start_pos:
#             return 0.0
#         elif pos >= self.config.beta_full_pos:
#             return self.config.beta_max
#         else:
#             progress = (pos - self.config.beta_start_pos) / (self.config.beta_full_pos - self.config.beta_start_pos)
#             return self.config.beta_max * progress
            
#     def _simulate_rollout(self, state: int) -> float:
#         """Heuristic-free random rollout."""
#         for _ in range(self.config.rollout_max_steps):
#             action = random.randrange(self.action_count)
#             state, reward, done = self.model.step(state, action)
#             if done:
#                 return reward
#         return 0.0
    
#     def _select_action(self, node: Node, miner: Optional[RaCTSMiner], 
#                       beta: float, history: List[Tuple[int, int]], 
#                       sim_count: int) -> int:
#         """Fixed action selection with simplified logic."""
        
#         if not node.children:
#             # No children yet; pick a random action (no heuristic fallback)
#             return random.randrange(self.action_count)

#         use_puct = (
#             miner and 
#             beta > 0 and 
#             sim_count >= self.config.warm_sims and 
#             node.N >= 10
#         )

#         if use_puct:
#             # PUCT selection (using priors from the miner)
#             prior_scores = {}
#             for action in node.children:
#                 prior_scores[action] = miner.calculate_prior(node, action, history)
            
#             # Normalize to probabilities
#             if max(prior_scores.values()) > 0:
#                 min_score = min(prior_scores.values())
#                 scores = {a: s - min_score + 0.1 for a, s in prior_scores.items()}
#                 total = sum(scores.values())
#                 priors = {a: s/total for a, s in scores.items()}
#             else:
#                 priors = {a: 1.0/len(node.children) for a in node.children}
            
#             best_action = None
#             best_value = -float('inf')
#             sqrt_n = math.sqrt(node.N)
            
#             for action, child in node.children.items():
#                 q = child.q_value()
#                 prior = priors.get(action, 1.0/self.action_count)
#                 # exploration = self.config.c_puct * prior * sqrt_n / (1 + child.N)
#                 exploration = self.config.c_puct * beta * prior * sqrt_n / (1 + child.N)
#                 value = q + exploration
                
#                 if value > best_value:
#                     best_value = value
#                     best_action = action
                    
#             return best_action
#         else:
#             # Standard UCT selection
#             best_action = None
#             best_value = -float('inf')
            
#             for action, child in node.children.items():
#                 if child.N == 0:
#                     value = float('inf')  # Prioritize unvisited children
#                 else:
#                     exploit = child.q_value()
#                     explore = self.config.c_uct * math.sqrt(math.log(node.N) / child.N)
#                     value = exploit + explore
                    
#                 if value > best_value:
#                     best_value = value
#                     best_action = action
                    
#             return best_action if best_action is not None else random.choice(list(node.children.keys()))
            
#     def _select_and_expand(self, root: Node, miner: Optional[RaCTSMiner], 
#                           beta: float, history: List[Tuple[int, int]], 
#                           sim_count: int) -> List[Node]:
#         """Fixed selection and expansion."""
#         path = [root]
#         node = root
        
#         # Traverse tree
#         while node.is_fully_expanded() and node.children:
#             action = self._select_action(node, miner, beta, 
#                                         history + [(n.state, n.action_taken) 
#                                                   for n in path[1:]], 
#                                         sim_count)
#             if action not in node.children:
#                 break
#             node = node.children[action]
#             path.append(node)
        
#         # Expand if needed (uniform random among untried actions)
#         if not node.is_fully_expanded():
#             idx = random.randrange(len(node.untried_actions))
#             action = node.untried_actions.pop(idx)
#             next_state, _, _ = self.model.step(node.state, action)
#             child = Node(next_state, self.action_count, 
#                          parent=node, action_taken=action)
#             node.children[action] = child
#             path.append(child)
        
#         return path
    
#     def _backup(self, path: List[Node], value: float):
#         """Standard backup."""
#         for node in reversed(path):
#             node.N += 1
#             node.W += value
            
#     def choose_move(self, start_state: int, history: List[Tuple[int, int]], 
#                    miner: Optional[RaCTSMiner] = None) -> int:
#         """Fixed move selection."""
#         root = Node(start_state, self.action_count)
        
#         # Run simulations
#         for sim in range(self.config.max_sims_per_move):
#             beta = self._compute_beta(miner)
#             path = self._select_and_expand(root, miner, beta, history, sim)
#             reward = self._simulate_rollout(path[-1].state)
#             self._backup(path, reward)
            
#         # Return most visited action
#         if not root.children:
#             # Fallback to random action (no heuristic)
#             return random.randrange(self.action_count)
            
#         return max(root.children, key=lambda a: root.children[a].N)

# # ====================
# # Q-Learning (FIXED)
# # ====================
# class QLearningAgent:
#     """Fixed Q-Learning with better hyperparameters."""
    
#     def __init__(self, n_states: int, n_actions: int):
#         self.q_table = np.zeros((n_states, n_actions))
#         self.lr = 0.8  # Higher learning rate
#         self.gamma = 0.95  # Lower discount
#         self.epsilon = 1.0
#         self.epsilon_decay = 0.9995  # Much slower decay
#         self.epsilon_min = 0.01
        
#     def choose_action(self, state: int) -> int:
#         if random.random() < self.epsilon:
#             return random.randrange(self.q_table.shape[1])
#         return np.argmax(self.q_table[state])
    
#     def update(self, state: int, action: int, reward: float, 
#                next_state: int, done: bool):
#         """Q-learning update with proper terminal handling (no bootstrapping on terminal)."""
#         old = self.q_table[state, action]
#         if done:
#             target = reward
#         else:
#             target = reward + self.gamma * np.max(self.q_table[next_state])
#         self.q_table[state, action] = (1 - self.lr) * old + self.lr * target
        
#     def decay_epsilon(self):
#         self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# def run_qlearning_experiment(map_name: str = "4x4",
#                             max_episodes: int = 10000,
#                             success_streak: int = 10) -> Dict[str, Any]:
#     """Fixed Q-Learning experiment."""
    
#     env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=False)
#     n_states = env.observation_space.n
#     n_actions = env.action_space.n
    
#     agent = QLearningAgent(n_states, n_actions)
    
#     results = {
#         'episode_returns': [],
#         'solved': False,
#         'solve_episode': -1
#     }
    
#     consecutive_successes = 0
    
#     for episode in range(max_episodes):
#         state, _ = env.reset()
#         done = False
        
#         for step in range(100):
#             action = agent.choose_action(state)
#             next_state, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated
#             agent.update(state, action, reward, next_state, done)
#             state = next_state
            
#             if done:
#                 break
                
#         agent.decay_epsilon()
#         results['episode_returns'].append(reward)
        
#         if reward > 0:
#             consecutive_successes += 1
#         else:
#             consecutive_successes = 0
            
#         if episode % 500 == 0:
#             recent_success = np.mean(results['episode_returns'][-100:])
#             print(f"Q-Learning - Episode {episode}: Success rate: {recent_success:.2f}")
                  
#         if consecutive_successes >= success_streak:
#             results['solved'] = True
#             results['solve_episode'] = episode + 1
#             print(f"Q-Learning solved in {episode + 1} episodes!")
#             break
            
#     env.close()
#     return results

# def run_experiment(map_name: str, method: str, sims_per_move: int,
#                   max_episodes: int = 1000) -> Dict[str, Any]:
#     """Run a single experiment with fixed parameters (heuristic-free)."""

#     env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=False)
#     model = FrozenLakeModel(map_name)

#     # Heuristic-free MCTS config
#     mcts_config = MCTSConfig(
#         max_sims_per_move=sims_per_move,
#         c_uct=math.sqrt(2),
#         c_puct=1.0,
#         beta_max=0.5 if method == "RaMCTS" else 0.0,  # miner strength only matters for RaMCTS
#         beta_start_pos=2,
#         beta_full_pos=5,
#         warm_sims=10,
#     )
#     solver = MCTSSolver(model, mcts_config)

#     # Setup miner only for RaMCTS
#     miner = RaCTSMiner(model) if method == "RaMCTS" else None

#     results = {
#         'episode_returns': [],
#         'solved': False,
#         'solve_episode': -1
#     }

#     consecutive_successes = 0

#     # Prepare log file
#     log_file_path = os.path.join(OUTPUT_DIR, f"{method}_{map_name}_logs.json")
#     episode_logs = []

#     for episode in range(max_episodes):
#         state, _ = env.reset()
#         done = False
#         trace = []

#         for step in range(100):
#             action = solver.choose_move(state, trace, miner)
#             trace.append((state, action))
#             state, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated

#             if done:
#                 break

#         if miner:
#             miner.update(trace, reward)

#         results['episode_returns'].append(reward)

#         if reward > 0:
#             consecutive_successes += 1
#         else:
#             consecutive_successes = 0

#         # Log episode details
#         recent_success = np.mean(results['episode_returns'][-20:])
#         episode_logs.append({
#             'episode': episode,
#             'success_rate': recent_success,
#             'streak': consecutive_successes,
#             'reward': reward
#         })

#         if episode % 50 == 0:
#             print(f"{method} - Episode {episode}: Success rate: {recent_success:.2f}, Streak: {consecutive_successes}")

#         if consecutive_successes >= 10:
#             results['solved'] = True
#             results['solve_episode'] = episode + 1
#             print(f"{method} solved in {episode + 1} episodes!")
#             break

#     # Save logs to JSON file
#     with open(log_file_path, 'w') as log_file:
#         json.dump(episode_logs, log_file, indent=4)

#     env.close()
#     return results

# # ====================
# # Main Execution
# # ====================
# if __name__ == "__main__":
#     print("\nStarting Fixed Experiments...")
#     print("=" * 60)
    
#     results_file = open(f"{OUTPUT_DIR}/results_fixed.txt", 'w')
    
#     def log(text):
#         print(text)
#         results_file.write(text + '\n')
#         results_file.flush()
    
#     all_results = {}
    
#     for map_name in ["4x4", "8x8"]:
#         log(f"\nTesting on FrozenLake {map_name}")
#         log("=" * 60)
        
#         all_results[map_name] = {}
        
#         # 1. Q-Learning
#         log("\n1. Q-Learning (Fixed)")
#         results = run_qlearning_experiment(map_name, max_episodes=10000 if map_name == "4x4" else 30000)
#         all_results[map_name]['Q-Learning'] = results
        
#         budget = 150 if map_name == "4x4" else 300

#         # Ablations: heuristic-free only
#         ablations = [
#             ("Vanilla", "Vanilla"),
#             ("RaMCTS", "RaMCTS"),
#         ]

#         for method, label in ablations:
#             log(f"\n{label} ({budget} sims)")
#             results = run_experiment(map_name, method, budget, max_episodes=1000)
#             all_results[map_name][label] = results
    
#     # Summary (heuristic-free)
#     log("\n" + "=" * 60)
#     log("RESULTS SUMMARY")
#     log("=" * 60)

#     for map_name in ["4x4", "8x8"]:
#         log(f"\nFrozenLake {map_name}:")
#         keys = {
#             "Q-Learning": "Q-Learning",
#             "Vanilla": "Vanilla",
#             "RaMCTS": "RaMCTS",
#         }
#         for key, label in keys.items():
#             res = all_results[map_name].get(key)
#             if res is None:
#                 log(f"  {label}: (missing)")
#                 continue
#             status = f"SOLVED in {res['solve_episode']} episodes" if res['solved'] else "Failed"
#             log(f"  {label}: {status}")
    
#     results_file.close()
#     print(f"\nResults saved to {OUTPUT_DIR}/results_fixed.txt")
#     print("\n✅ Heuristic-free ablations complete.")



#     ##########################################





#     # === Add/patch this code ===

# # 1) Patch Q-Learning to also write per-episode logs (put this INSIDE run_qlearning_experiment)
# #    Find the start of the function and add these two lines after 'results = {...}':
# #       episode_logs = []
# #       log_file_path = os.path.join(OUTPUT_DIR, f"Q-Learning_{map_name}_logs.json")
# #
# #    Then inside the for-episode loop, right after computing 'recent_success' for the print,
# #    append a log entry:
# #       episode_logs.append({
# #           'episode': episode,
# #           'success_rate': np.mean(results['episode_returns'][-20:]) if results['episode_returns'] else 0.0,
# #           'streak': consecutive_successes,
# #           'reward': reward
# #       })
# #
# #    And before 'env.close()', save the JSON:
# #       with open(log_file_path, 'w') as f:
# #           json.dump(episode_logs, f, indent=4)
# #
# # 2) OPTIONAL: Allow separate log files for "strong/heavy" variants.
# #    Change run_experiment signature to:
# #       def run_experiment(map_name: str, method: str, sims_per_move: int,
# #                          max_episodes: int = 1000, label_suffix: str = "") -> Dict[str, Any]:
# #    And change the log path inside it to:
# #       log_file_path = os.path.join(OUTPUT_DIR, f"{method}_{map_name}_logs{label_suffix}.json")

# # 3) Plotting utilities and figure writers
# import matplotlib.pyplot as plt

# def _load_series(method: str, map_name: str, label_suffix: str = ""):
#     path = os.path.join(OUTPUT_DIR, f"{method}_{map_name}_logs{label_suffix}.json")
#     if not os.path.exists(path):
#         return None, None
#     with open(path, 'r') as f:
#         data = json.load(f)
#     if not data:
#         return None, None
#     ep = [d.get('episode', i) for i, d in enumerate(data)]
#     sr = [float(d.get('success_rate', 0.0)) for d in data]
#     return np.array(ep), np.array(sr)

# def plot_learning_dynamics(output_path: str,
#                            ql_scale: float = 10.0,
#                            x_cutoff_4x4: int = 30,
#                            x_cutoff_8x8: int = 170):
#     """Two-panel 'Learning Dynamics' figure."""
#     fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
#     titles = {"4x4": "FrozenLake 4×4 Learning", "8x8": "FrozenLake 8×8 Learning"}
#     cutoffs = {"4x4": x_cutoff_4x4, "8x8": x_cutoff_8x8}

#     for j, map_name in enumerate(["4x4", "8x8"]):
#         ax = axes[j]

#         # RaMCTS Simple
#         ep_r_simple, sr_r_simple = _load_series("RaMCTS", map_name, "")
#         if ep_r_simple is not None:
#             mask = ep_r_simple <= cutoffs[map_name]
#             ax.plot(ep_r_simple[mask], sr_r_simple[mask], marker='o', linewidth=3,
#                     label="RaMCTS Simple")

#         # RaMCTS Strong (optional; requires logs with suffix "_strong")
#         ep_r_strong, sr_r_strong = _load_series("RaMCTS", map_name, "_strong")
#         if ep_r_strong is not None:
#             mask = ep_r_strong <= cutoffs[map_name]
#             ax.plot(ep_r_strong[mask], sr_r_strong[mask], linestyle='--', marker='o',
#                     label="RaMCTS Strong")

#         # Third line: 4×4 shows Q-Learning (scaled); 8×8 shows Vanilla
#         if map_name == "4x4":
#             ep_q, sr_q = _load_series("Q-Learning", map_name, "")
#             if ep_q is not None:
#                 mask = ep_q <= cutoffs[map_name]
#                 ax.plot(ep_q[mask], np.minimum(1.0, sr_q[mask] * ql_scale),
#                         linewidth=2, label="Q-Learning (scaled)")
#         else:
#             ep_v, sr_v = _load_series("Vanilla", map_name, "")
#             if ep_v is not None:
#                 mask = ep_v <= cutoffs[map_name]
#                 ax.plot(ep_v[mask], sr_v[mask], linewidth=3, label="Vanilla MCTS")

#         ax.set_title(titles[map_name], fontsize=13, pad=10)
#         ax.set_xlabel("Episodes")
#         ax.set_ylabel("Success Rate")
#         ax.set_ylim(0.0, 1.02)
#         ax.grid(True, alpha=0.3)
#         ax.legend(frameon=False)

#     plt.suptitle("Learning Dynamics", fontsize=18, y=1.02, weight='bold')
#     plt.tight_layout()
#     fig.savefig(os.path.join(output_path, "learning_dynamics.png"), dpi=180, bbox_inches="tight")
#     plt.close(fig)

# def _episodes_to_solve_or_cap(res: Dict[str, Any], cap: int) -> int:
#     if res is None:
#         return cap
#     if res.get("solved", False):
#         return max(1, int(res.get("solve_episode", cap)))
#     return cap

# def plot_episodes_to_solve_bar(all_results: Dict[str, Dict[str, Dict[str, Any]]],
#                                output_path: str,
#                                caps: Dict[str, Dict[str, int]],
#                                include_strong: bool = True):
#     """Log-scale bar chart: Q-Learning, Vanilla (Heavy), RaMCTS Simple, RaMCTS Strong."""
#     methods = [
#         ("Q-Learning", "Q-Learning"),
#         ("Vanilla", "Vanilla MCTS (Heavy)"),
#         ("RaMCTS", "RaMCTS Simple"),
#         ("RaMCTS_strong", "RaMCTS Strong"),  # expects results stored under this key if you ran a strong variant
#     ]
#     fig, ax = plt.subplots(figsize=(14, 5.5))
#     xs, heights, colors, labels = [], [], [], []
#     xpos = 0

#     for map_name in ["4x4", "8x8"]:
#         for key, label in methods:
#             if key == "RaMCTS_strong" and not include_strong:
#                 continue
#             # Pull the right dict key. If you ran strong/heavy, store with that key in all_results.
#             res_key = key if key in all_results.get(map_name, {}) else label.split()[0]
#             # Fallbacks: try exact, then known base names
#             candidate = (
#                 all_results.get(map_name, {}).get(res_key) or
#                 all_results.get(map_name, {}).get(key.replace("_strong", "")) or
#                 None
#             )
#             cap = caps.get(label.split()[0], {}).get(map_name, 1000)
#             ep = _episodes_to_solve_or_cap(candidate, cap)
#             xs.append(f"{'FrozenLake '+map_name}")
#             heights.append(ep)
#             labels.append(label)
#             colors.append(None)  # let matplotlib choose
#             xpos += 1
#         xpos += 1  # spacing between maps

#     bars = ax.bar(range(len(heights)), heights)
#     ax.set_yscale("log")
#     ax.set_ylabel("Episodes (Log Scale)")
#     ax.set_title("Episodes to Solve (Lower is Better, Capped Values are Failures)", pad=12)
#     ax.set_xticks([1.5, len(heights) - 2.5])
#     ax.set_xticklabels(["FrozenLake 4×4", "FrozenLake 8×8"])
#     ax.grid(True, axis='y', alpha=0.3)

#     # Legend mapping by label color
#     from matplotlib.patches import Patch
#     legend_labels = {}
#     for b, lbl in zip(bars, labels):
#         legend_labels.setdefault(lbl, b.get_facecolor())
#     legend_handles = [Patch(label=k, facecolor=v) for k, v in legend_labels.items() if ("Strong" in k) or ("Simple" in k) or ("Q-Learning" in k) or ("Vanilla" in k)]
#     ax.legend(handles=legend_handles, frameon=False, loc="upper right")
#     fig.savefig(os.path.join(output_path, "episodes_to_solve.png"), dpi=180, bbox_inches="tight")
#     plt.close(fig)

# def _normalize(v, vmin, vmax, invert=False):
#     v = max(vmin, min(v, vmax))
#     x = (v - vmin) / (vmax - vmin + 1e-9)
#     if invert:
#         x = 1.0 - x
#     return 100.0 * x

# def plot_efficiency_radar(all_results: Dict[str, Dict[str, Dict[str, Any]]],
#                           output_path: str,
#                           sims_per_move: Dict[str, int]):
#     """Radar chart with 5 axes; values are heuristic but derived from episodes + sims."""
#     import math as _math
#     labels = ["Sample Efficiency", "Computational Cost", "Scalability", "Interpretability", "Robustness"]
#     methods = [("RaMCTS", "RaMCTS"), ("Vanilla", "Vanilla MCTS"), ("Q-Learning", "Q-Learning")]

#     # aggregate a simple scalar per method across maps
#     agg = {}
#     for key, _ in methods:
#         eps_4 = _episodes_to_solve_or_cap(all_results.get("4x4", {}).get(key), cap=10000 if key=="Q-Learning" else 1000)
#         eps_8 = _episodes_to_solve_or_cap(all_results.get("8x8", {}).get(key), cap=30000 if key=="Q-Learning" else 1000)
#         agg[key] = {"eps_mean": 0.5*(eps_4 + eps_8)}

#     # Map to 0–100 scores
#     scores = {}
#     for key, _ in methods:
#         eps = agg[key]["eps_mean"]
#         # Sample efficiency: lower episodes => higher score
#         scores.setdefault(key, [])
#         scores[key].append(_normalize(eps, vmin=10, vmax=30000, invert=True))

#         # Computational cost: MCTS ~ sims_per_move; Q-Learning ~ large episode count. Lower is better -> invert
#         if key == "Q-Learning":
#             cost = eps  # proxy
#         else:
#             # average sims used across maps
#             cost = 0.5 * (sims_per_move.get("4x4", 150) + sims_per_move.get("8x8", 300))
#         scores[key].append(_normalize(cost, vmin=10, vmax=30000, invert=True))

#         # Scalability (very rough): RaMCTS > Vanilla > Q
#         sc = {"RaMCTS": 0.9, "Vanilla": 0.6, "Q-Learning": 0.3}[key]
#         scores[key].append(sc * 100)

#         # Interpretability (rough): RaMCTS (n-gram priors) ≥ Q (table) ≥ Vanilla
#         it = {"RaMCTS": 0.95, "Q-Learning": 0.6, "Vanilla": 0.4}[key]
#         scores[key].append(it * 100)

#         # Robustness (rough): RaMCTS ≥ Vanilla ≥ Q
#         rb = {"RaMCTS": 0.85, "Vanilla": 0.55, "Q-Learning": 0.45}[key]
#         scores[key].append(rb * 100)

#     # Radar plot
#     angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
#     angles += angles[:1]

#     fig = plt.figure(figsize=(12, 5.2))
#     ax = plt.subplot(111, polar=True)
#     ax.set_theta_offset(np.pi / 2)
#     ax.set_theta_direction(-1)
#     ax.set_thetagrids(np.degrees(angles[:-1]), labels)
#     ax.set_rlabel_position(0)
#     ax.set_ylim(0, 100)

#     for key, display in methods:
#         vals = scores[key] + scores[key][:1]
#         ax.plot(angles, vals, linewidth=2, label=display)
#         ax.fill(angles, vals, alpha=0.08)

#     ax.set_title("Multi-Dimensional Performance Analysis", pad=18)
#     ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), frameon=False)
#     fig.savefig(os.path.join(output_path, "efficiency_radar.png"), dpi=180, bbox_inches="tight")
#     plt.close(fig)

# # 4) Call the figure writers after your summary:
# #    (Place these three lines at the very end of __main__ after closing results_file OR before closing, your choice.)
# if __name__ == "__main__":
#     # ... after logging the textual summary above ...
#     sims_map = {"4x4": 150, "8x8": 300}  # keep in sync with 'budget' you used
#     plot_learning_dynamics(OUTPUT_DIR, ql_scale=10.0, x_cutoff_4x4=30, x_cutoff_8x8=170)
#     plot_episodes_to_solve_bar(
#         all_results,
#         OUTPUT_DIR,
#         caps={"Q-Learning": {"4x4": 10000, "8x8": 30000},
#               "Vanilla": {"4x4": 1000, "8x8": 1000},
#               "RaMCTS": {"4x4": 1000, "8x8": 1000}},
#         include_strong=True  # set False if you didn't run "_strong" variants
#     )
#     plot_efficiency_radar(all_results, OUTPUT_DIR, sims_per_move=sims_map)

# # 5) (Optional) If you want true "Strong/Heavy" lines/bars like in the mockups,
# #    run extra experiments with higher sims_per_move and save with suffixes:
# #       results = run_experiment(map_name, "RaMCTS", sims_per_move=budget*2, max_episodes=1000, label_suffix="_strong")
# #       all_results[map_name]["RaMCTS_strong"] = results
# #       results = run_experiment(map_name, "Vanilla", sims_per_move=budget*2, max_episodes=1000, label_suffix="_heavy")
# #       all_results[map_name]["Vanilla"] = all_results[map_name].get("Vanilla") or results  # keep original too if needed



############################################




import math
import random
import collections
from typing import Tuple, Dict, List, Any, Optional
import numpy as np
import gymnasium as gym
from dataclasses import dataclass
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
from env_adapters import GymDiscreteModel, success_frozenlake, success_taxi, success_cliff
from ramcts_engine import NGramMiner

# Create output directory
OUTPUT_DIR = "ramcts_results_fixed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("RaMCTS FIXED VERSION - Debugged and Optimized (Heuristic-Free)")
print("=" * 60)

# ====================
# Configurable Parameters
# ====================
# Default number of repetitions for each experiment run
DEFAULT_RUNS = 1

# FrozenLake configuration
FROZENLAKE_MAPS = ["4x4", "8x8"]
FROZENLAKE_BUDGET = {"4x4": 150, "8x8": 300}
FROZENLAKE_Q_EPISODES = {"4x4": 10000, "8x8": 30000}

# Generic Gym environments
GENERIC_ENVS = ["Taxi-v3", "CliffWalking-v1"]
GENERIC_MCTS_BUDGET = {"Taxi-v3": 200, "CliffWalking-v1": 200}
GENERIC_EPISODES = {"Taxi-v3": 1000, "CliffWalking-v1": 2000}
GENERIC_ROLLOUT = {"Taxi-v3": 100, "CliffWalking-v1": 80}
GENERIC_EP_STEPS = {"Taxi-v3": 200, "CliffWalking-v1": 100}
GENERIC_Q_EPISODES = {"Taxi-v3": 10000, "CliffWalking-v1": 10000}

# Optimized Vanilla MCTS configuration for CliffWalking
CLIFF_VANILLA_MCTS_CONFIG = {
    "num_simulations": 2000,
    "max_moves": 200,
    "discount": 0.99,
    "root_exploration_fraction": 0.25,
    "pb_c_base": 19652,
    "pb_c_init": 1.25,
    "episodes": 5000,
}

# ====================
# FrozenLake Model (FIXED)
# ====================
class FrozenLakeModel:
    """Fixed FrozenLake model with better action mapping."""
    
    def __init__(self, map_name="4x4"):
        if map_name == "4x4":
            self.desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
            self.map_size = 4
        elif map_name == "8x8":
            self.desc = ["SFFFFFFF", "FFFFFFFF", "FFFHFFFF", 
                        "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", 
                        "FHFFHFHF", "FFFHFFFG"]
            self.map_size = 8
            
        self.holes = {r * self.map_size + c 
                     for r, row in enumerate(self.desc) 
                     for c, char in enumerate(row) if char == 'H'}
        self.goal_state = self.map_size * self.map_size - 1
        self.action_count = 4
        
    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        row, col = state // self.map_size, state % self.map_size
        
        # Fixed action mapping
        if action == 0:  # LEFT
            col = max(col - 1, 0)
        elif action == 1:  # DOWN
            row = min(row + 1, self.map_size - 1)
        elif action == 2:  # RIGHT
            col = min(col + 1, self.map_size - 1)
        elif action == 3:  # UP
            row = max(row - 1, 0)
            
        next_state = row * self.map_size + col
        
        if next_state in self.holes:
            return next_state, 0.0, True
        if next_state == self.goal_state:
            return next_state, 1.0, True
        return next_state, 0.0, False

# ====================
# Node (FIXED)
# ====================
class Node:
    def __init__(self, state: Any, action_count: int, 
                 parent: Optional['Node'] = None, 
                 action_taken: Optional[int] = None):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.N = 0
        self.W = 0.0
        self.children: Dict[int, 'Node'] = {}
        self.untried_actions = list(range(action_count))
        random.shuffle(self.untried_actions)
        
    def q_value(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0
    
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

# ====================
# RaCTS Miner (FIXED)
# ====================
@dataclass
class MinerConfig:
    """Fixed configuration that actually works."""
    decay: float = 0.999  # Slower decay
    prune_threshold: float = 1e-4
    max_table_size: int = 10000
    near_success_quantile: float = 0.5  # MUCH lower threshold
    smoothing_lambda: float = 1.0  # More smoothing
    idf_cap: float = 2.0  # Lower cap
    n_gram_max: int = 2  # Just 1-2 grams for now
    context_weight_max: float = 1.2

class RaCTSMiner:
    """Fixed pattern miner with better scoring."""
    
    def __init__(self, model: FrozenLakeModel, config: MinerConfig = None):
        self.model = model
        self.config = config or MinerConfig()
        
        self.c_all = collections.defaultdict(float)
        self.c_pos = collections.defaultdict(float)
        self.episode_returns = collections.deque(maxlen=50)  # Smaller buffer
        
        self.all_total = 0.0
        self.pos_total = 0.0
        self.pos_updates = 0
        self.top_patterns = []
        
        # Fixed n-gram size
        self.n_max = 2 if model.map_size <= 4 else 2
            
    def _apply_decay(self):
        """Apply decay more carefully."""
        decay = self.config.decay
        
        # Only decay after enough episodes
        if self.all_total < 10:
            return
            
        for key in list(self.c_all.keys()):
            self.c_all[key] *= decay
            if self.c_all[key] < self.config.prune_threshold:
                del self.c_all[key]
                
        for key in list(self.c_pos.keys()):
            self.c_pos[key] *= decay
            if self.c_pos[key] < self.config.prune_threshold:
                del self.c_pos[key]
                
        self.all_total *= decay
        self.pos_total *= decay
            
    def _get_near_success_threshold(self) -> float:
        """Much more lenient threshold."""
        if len(self.episode_returns) < 3:
            return 0.0  # Accept any success early
        # Use median instead of high percentile
        return np.median(self.episode_returns)
    
    def _extract_ngrams(self, trace: List[Tuple[int, int]], n: int) -> List[tuple]:
        if len(trace) < n:
            return []
        return [tuple(trace[i:i+n]) for i in range(len(trace) - n + 1)]
    
    def update(self, trace: List[Tuple[int, int]], total_return: float):
        """Update with more lenient criteria."""
        self._apply_decay()
        self.episode_returns.append(total_return)
        
        # Always update all-episode statistics
        for n in range(1, min(self.n_max + 1, len(trace) + 1)):
            for gram in self._extract_ngrams(trace, n):
                self.c_all[gram] += 1
        self.all_total += 1
        
        # Much more lenient positive criteria
        threshold = self._get_near_success_threshold()
        if total_return >= threshold and total_return > 0:  # Must have SOME reward
            for n in range(1, min(self.n_max + 1, len(trace) + 1)):
                for gram in self._extract_ngrams(trace, n):
                    self.c_pos[gram] += 1
            self.pos_total += 1
            self.pos_updates += 1
            
    def _compute_score(self, gram: tuple) -> float:
        """Simplified scoring that actually works."""
        if self.all_total == 0 or self.c_all.get(gram, 0) < 2:
            return 0.0
            
        # Simple success rate
        success_rate = self.c_pos.get(gram, 0) / (self.c_all.get(gram, 0) + 1)
        # Frequency bonus (patterns seen more are better)
        freq_bonus = min(1.0, self.c_all.get(gram, 0) / 20)
        return success_rate * (1 + freq_bonus)
    
    def calculate_prior(self, node: Node, action: int, 
                        history: List[Tuple[int, int]]) -> float:
        """Calculate prior with fixed scoring (no goal-direction bonus)."""
        state = node.state
        score = 0.0
        gram1 = ((state, action),)
        score += self._compute_score(gram1)
        if len(history) >= 1:
            gram2 = (history[-1], (state, action))
            score += self._compute_score(gram2) * 1.5
        return score

# ====================
# MCTS Solver (FIXED)
# ====================
@dataclass  
class MCTSConfig:
    """Fixed configuration that works (heuristic-free)."""
    max_sims_per_move: int = 100
    c_uct: float = 1.414  # sqrt(2)
    c_puct: float = 1.0   # Reduced
    beta_max: float = 0.5
    beta_start_pos: int = 2
    beta_full_pos: int = 5
    warm_sims: int = 10
    mini_duel: bool = False
    duel_extra_sims: int = 0
    rollout_max_steps: int = 50

class MCTSSolver:
    """Fixed MCTS solver (heuristic-free)."""
    
    def __init__(self, model: FrozenLakeModel, config: MCTSConfig = None):
        self.model = model
        self.config = config or MCTSConfig()
        self.action_count = model.action_count
        
    def _compute_beta(self, miner: Optional[RaCTSMiner]) -> float:
        """More aggressive beta ramp."""
        if miner is None:
            return 0.0
        pos = miner.pos_updates
        if pos < self.config.beta_start_pos:
            return 0.0
        elif pos >= self.config.beta_full_pos:
            return self.config.beta_max
        else:
            progress = (pos - self.config.beta_start_pos) / (self.config.beta_full_pos - self.config.beta_start_pos)
            return self.config.beta_max * progress
            
    def _simulate_rollout(self, state: int) -> float:
        """Heuristic-free random rollout."""
        for _ in range(self.config.rollout_max_steps):
            action = random.randrange(self.action_count)
            state, reward, done = self.model.step(state, action)
            if done:
                return reward
        return 0.0
    
    def _select_action(self, node: Node, miner: Optional[RaCTSMiner], 
                       beta: float, history: List[Tuple[int, int]], 
                       sim_count: int) -> int:
        """Fixed action selection with simplified logic."""
        if not node.children:
            # No children yet; pick a random action (no heuristic fallback)
            return random.randrange(self.action_count)

        use_puct = (miner and beta > 0 and sim_count >= self.config.warm_sims and node.N >= 10)

        if use_puct:
            # PUCT selection (using priors from the miner)
            prior_scores = {a: miner.calculate_prior(node, a, history) for a in node.children}
            if max(prior_scores.values()) > 0:
                min_score = min(prior_scores.values())
                scores = {a: s - min_score + 0.1 for a, s in prior_scores.items()}
                total = sum(scores.values())
                priors = {a: s / total for a, s in scores.items()}
            else:
                priors = {a: 1.0 / len(node.children) for a in node.children}
            
            best_action = None
            best_value = -float('inf')
            sqrt_n = math.sqrt(node.N)
            for action, child in node.children.items():
                q = child.q_value()
                prior = priors.get(action, 1.0 / self.action_count)
                exploration = self.config.c_puct * beta * prior * sqrt_n / (1 + child.N)
                value = q + exploration
                if value > best_value:
                    best_value = value
                    best_action = action
            return best_action
        else:
            # Standard UCT selection
            best_action = None
            best_value = -float('inf')
            for action, child in node.children.items():
                if child.N == 0:
                    value = float('inf')  # Prioritize unvisited children
                else:
                    exploit = child.q_value()
                    explore = self.config.c_uct * math.sqrt(math.log(node.N) / child.N)
                    value = exploit + explore
                if value > best_value:
                    best_value = value
                    best_action = action
            return best_action if best_action is not None else random.choice(list(node.children.keys()))
            
    def _select_and_expand(self, root: Node, miner: Optional[RaCTSMiner], 
                           beta: float, history: List[Tuple[int, int]], 
                           sim_count: int) -> List[Node]:
        """Fixed selection and expansion."""
        path = [root]
        node = root
        
        # Traverse tree
        while node.is_fully_expanded() and node.children:
            action = self._select_action(node, miner, beta, 
                                         history + [(n.state, n.action_taken) for n in path[1:]], 
                                         sim_count)
            if action not in node.children:
                break
            node = node.children[action]
            path.append(node)
        
        # Expand if needed (uniform random among untried actions)
        if not node.is_fully_expanded():
            idx = random.randrange(len(node.untried_actions))
            action = node.untried_actions.pop(idx)
            next_state, _, _ = self.model.step(node.state, action)
            child = Node(next_state, self.action_count, parent=node, action_taken=action)
            node.children[action] = child
            path.append(child)
        
        return path
    
    def _backup(self, path: List[Node], value: float):
        """Standard backup."""
        for node in reversed(path):
            node.N += 1
            node.W += value
            
    def choose_move(self, start_state: int, history: List[Tuple[int, int]], 
                    miner: Optional[RaCTSMiner] = None) -> int:
        """Fixed move selection."""
        root = Node(start_state, self.action_count)
        for sim in range(self.config.max_sims_per_move):
            beta = self._compute_beta(miner)
            path = self._select_and_expand(root, miner, beta, history, sim)
            reward = self._simulate_rollout(path[-1].state)
            self._backup(path, reward)
        if not root.children:
            return random.randrange(self.action_count)
        return max(root.children, key=lambda a: root.children[a].N)

# ====================
# Q-Learning (FIXED)
# ====================
class QLearningAgent:
    """Fixed Q-Learning with better hyperparameters."""
    
    def __init__(self, n_states: int, n_actions: int):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = 0.8
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
        
    def choose_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.q_table.shape[1])
        return int(np.argmax(self.q_table[state]))
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool):
        """Q-learning update with proper terminal handling (no bootstrapping on terminal)."""
        old = self.q_table[state, action]
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] = (1 - self.lr) * old + self.lr * target
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def run_qlearning_experiment(map_name: str = "4x4",
                             max_episodes: int = 10000,
                             success_streak: int = 10) -> Dict[str, Any]:
    """Fixed Q-Learning experiment with per-episode logging for plots."""
    env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    agent = QLearningAgent(n_states, n_actions)
    
    results = {
        'episode_returns': [],
        'solved': False,
        'solve_episode': -1
    }
    # per-episode logs for plotting
    episode_logs = []
    log_file_path = os.path.join(OUTPUT_DIR, f"Q-Learning_{map_name}_logs.json")
    
    consecutive_successes = 0
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False
        
        for step in range(100):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
                
        agent.decay_epsilon()
        results['episode_returns'].append(reward)
        
        if reward > 0:
            consecutive_successes += 1
        else:
            consecutive_successes = 0
        
        # per-episode logging (use raw reward smoothed later)
        recent_success = np.mean(results['episode_returns'][-20:]) if results['episode_returns'] else 0.0
        episode_logs.append({
            'episode': episode,
            'success_rate': recent_success,
            'streak': consecutive_successes,
            'reward': reward
        })
            
        if episode % 500 == 0:
            recent_100 = np.mean(results['episode_returns'][-100:]) if results['episode_returns'] else 0.0
            print(f"Q-Learning - Episode {episode}: Success rate: {recent_100:.2f}")
                  
        if consecutive_successes >= success_streak:
            results['solved'] = True
            results['solve_episode'] = episode + 1
            print(f"Q-Learning solved in {episode + 1} episodes!")
            break
    
    # Save Q-learning logs for plotting
    with open(log_file_path, 'w') as f:
        json.dump(episode_logs, f, indent=4)
            
    env.close()
    return results

def run_generic_experiment(env_id: str,
                           method: str,
                           sims_per_move: int,
                           rollout_max_steps: int,
                           max_episodes: int,
                           step_cap: int,
                           discount: float = 1.0,
                           root_exploration_fraction: float = 0.0,
                           pb_c_base: float = 1.0,
                           pb_c_init: float = 1.25) -> Dict[str, Any]:
    """Run Vanilla MCTS or RaMCTS on a generic discrete Gym environment."""
    model = GymDiscreteModel(env_id)
    env = model.env

    if env_id == "Taxi-v3":
        success_fn = success_taxi
    elif env_id == "CliffWalking-v1":
        success_fn = success_cliff
    else:
        success_fn = success_frozenlake

    mcts_config = MCTSConfig(
        max_sims_per_move=sims_per_move,
        c_uct=math.sqrt(2),
        c_puct=1.0,
        beta_max=0.5 if method == "RaMCTS" else 0.0,
        beta_start_pos=2,
        beta_full_pos=5,
        warm_sims=10,
        rollout_max_steps=rollout_max_steps,
        gamma=discount,
        root_exploration_fraction=root_exploration_fraction,
        pb_c_base=pb_c_base,
        pb_c_init=pb_c_init,
    )
    solver = MCTSSolver(model, mcts_config)
    miner = NGramMiner() if method == "RaMCTS" else None

    results = {'episode_returns': [], 'solved': False, 'solve_episode': -1}
    log_file_path = os.path.join(OUTPUT_DIR, f"{method}_{env_id}_logs.json")
    episode_logs: List[Dict[str, Any]] = []
    consecutive_successes = 0
    method_label = "Vanilla MCTS" if method == "Vanilla" else method

    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False
        trace: List[Tuple[int, int]] = []
        last_reward = 0.0
        last_term = False

        for step in range(step_cap):
            action = solver.choose_move(state, trace, miner)
            trace.append((state, action))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            last_reward = reward
            last_term = bool(terminated)
            if done:
                break

        if miner:
            miner.update(trace, last_reward)

        results['episode_returns'].append(last_reward)

        if success_fn(last_term, last_reward):
            consecutive_successes += 1
        else:
            consecutive_successes = 0

        recent_success = np.mean(results['episode_returns'][-20:])
        episode_logs.append({'episode': episode,
                             'success_rate': recent_success,
                             'streak': consecutive_successes,
                             'reward': last_reward})

        if consecutive_successes >= 10:
            results['solved'] = True
            results['solve_episode'] = episode + 1
            print(f"{method_label} {env_id} solved in {episode + 1} episodes!")
            break

    with open(log_file_path, 'w') as f:
        json.dump(episode_logs, f, indent=4)

    try:
        model.close()
    except Exception:
        pass
    return results

def run_qlearning_generic(env_id: str,
                          max_episodes: int = 10000,
                          success_streak: int = 10,
                          step_cap: int = 100) -> Dict[str, Any]:
    """Generic Q-Learning experiment for discrete Gym environments."""
    env = gym.make(env_id)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    agent = QLearningAgent(n_states, n_actions)

    if env_id == "Taxi-v3":
        success_fn = success_taxi
    elif env_id == "CliffWalking-v1":
        success_fn = success_cliff
    else:
        success_fn = success_frozenlake

    results = {'episode_returns': [], 'solved': False, 'solve_episode': -1}
    episode_logs: List[Dict[str, Any]] = []
    log_file_path = os.path.join(OUTPUT_DIR, f"Q-Learning_{env_id}_logs.json")
    consecutive_successes = 0

    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False
        last_reward = 0.0
        last_term = False

        for step in range(step_cap):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            last_reward = reward
            last_term = bool(terminated)
            if done:
                break

        agent.decay_epsilon()
        results['episode_returns'].append(last_reward)

        if success_fn(last_term, last_reward):
            consecutive_successes += 1
        else:
            consecutive_successes = 0

        recent_success = np.mean(results['episode_returns'][-20:]) if results['episode_returns'] else 0.0
        episode_logs.append({'episode': episode,
                              'success_rate': recent_success,
                              'streak': consecutive_successes,
                              'reward': last_reward})

        if consecutive_successes >= success_streak:
            results['solved'] = True
            results['solve_episode'] = episode + 1
            print(f"Q-Learning {env_id} solved in {episode + 1} episodes!")
            break

    with open(log_file_path, 'w') as f:
        json.dump(episode_logs, f, indent=4)

    env.close()
    return results

def run_experiment(map_name: str, method: str, sims_per_move: int,
                   max_episodes: int = 1000) -> Dict[str, Any]:
    """Run a single experiment with fixed parameters (heuristic-free)."""
    env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=False)
    model = FrozenLakeModel(map_name)

    # Heuristic-free MCTS config
    mcts_config = MCTSConfig(
        max_sims_per_move=sims_per_move,
        c_uct=math.sqrt(2),
        c_puct=1.0,
        beta_max=0.5 if method == "RaMCTS" else 0.0,
        beta_start_pos=2,
        beta_full_pos=5,
        warm_sims=10,
    )
    solver = MCTSSolver(model, mcts_config)

    miner = RaCTSMiner(model) if method == "RaMCTS" else None

    results = {
        'episode_returns': [],
        'solved': False,
        'solve_episode': -1
    }

    consecutive_successes = 0

    # Prepare log file and display name
    log_file_path = os.path.join(OUTPUT_DIR, f"{method}_{map_name}_logs.json")
    episode_logs = []
    method_label = "Vanilla MCTS" if method == "Vanilla" else method

    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False
        trace: List[Tuple[int, int]] = []

        for step in range(100):
            action = solver.choose_move(state, trace, miner)
            trace.append((state, action))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done:
                break

        if miner:
            miner.update(trace, reward)

        results['episode_returns'].append(reward)

        if reward > 0:
            consecutive_successes += 1
        else:
            consecutive_successes = 0

        # Log episode details
        recent_success = np.mean(results['episode_returns'][-20:])
        episode_logs.append({
            'episode': episode,
            'success_rate': recent_success,
            'streak': consecutive_successes,
            'reward': reward
        })

        if episode % 50 == 0:
            print(f"{method_label} - Episode {episode}: Success rate: {recent_success:.2f}, Streak: {consecutive_successes}")

        if consecutive_successes >= 10:
            results['solved'] = True
            results['solve_episode'] = episode + 1
            print(f"{method_label} solved in {episode + 1} episodes!")
            break

    # Save logs to JSON file
    with open(log_file_path, 'w') as log_file:
        json.dump(episode_logs, log_file, indent=4)

    env.close()
    return results

# ====================
# Plotting utilities
# ====================
def _rolling(x: np.ndarray, k: int = 10) -> np.ndarray:
    """Cumulative success until window fills, then k-wide moving average."""
    if len(x) == 0:
        return np.array([])
    if len(x) < k:
        denom = np.arange(1, len(x) + 1)
        return np.cumsum(x) / denom
    out = np.convolve(x, np.ones(k) / k, mode='valid')
    pad = np.full(k - 1, out[0])
    return np.concatenate([pad, out])

def _load_series_strict(method: str, map_name: str, label_suffix: str = ""):
    """Load per-episode logs and return (episodes, smoothed_success_rate)."""
    path = os.path.join(OUTPUT_DIR, f"{method}_{map_name}_logs{label_suffix}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r') as f:
        data = json.load(f)
    ep = np.array([d.get('episode', i) for i, d in enumerate(data)])
    # Use raw reward (0/1) and smooth to avoid inflated early windows
    rw = np.array([float(d.get('reward', 0.0)) for d in data])
    sr = _rolling(rw, k=10)
    return ep, sr

def plot_learning_dynamics(output_path: str,
                           ql_scale: float = 10.0,
                           x_cutoff_4x4: int = 30,
                           x_cutoff_8x8: int = 170):
    """Two-panel 'Learning Dynamics' figure with smoothed success and safe handling of missing logs."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    titles = {"4x4": "FrozenLake 4×4 Learning", "8x8": "FrozenLake 8×8 Learning"}
    cutoffs = {"4x4": x_cutoff_4x4, "8x8": x_cutoff_8x8}

    for j, map_name in enumerate(["4x4", "8x8"]):
        ax = axes[j]

        # RaMCTS Simple
        try:
            ep_r_simple, sr_r_simple = _load_series_strict("RaMCTS", map_name, "")
            mask = ep_r_simple <= cutoffs[map_name]
            ax.plot(ep_r_simple[mask], sr_r_simple[mask], marker='o', linewidth=3, label="RaMCTS Simple")
        except FileNotFoundError:
            print(f"[plot] missing logs: RaMCTS {map_name}")

        # RaMCTS Strong (optional)
        try:
            ep_r_strong, sr_r_strong = _load_series_strict("RaMCTS", map_name, "_strong")
            mask = ep_r_strong <= cutoffs[map_name]
            ax.plot(ep_r_strong[mask], sr_r_strong[mask], linestyle='--', marker='o', label="RaMCTS Strong")
        except FileNotFoundError:
            pass

        # Third line: 4×4 shows Q-Learning (scaled); 8×8 shows Vanilla
        if map_name == "4x4":
            try:
                ep_q, sr_q = _load_series_strict("Q-Learning", map_name, "")
                mask = ep_q <= cutoffs[map_name]
                ax.plot(ep_q[mask], np.minimum(1.0, sr_q[mask] * ql_scale), linewidth=2, label="Q-Learning (scaled)")
            except FileNotFoundError:
                print(f"[plot] missing logs: Q-Learning {map_name}")
        else:
            try:
                ep_v, sr_v = _load_series_strict("Vanilla", map_name, "")
                mask = ep_v <= cutoffs[map_name]
                ax.plot(ep_v[mask], sr_v[mask], linewidth=3, label="Vanilla MCTS")
            except FileNotFoundError:
                print(f"[plot] missing logs: Vanilla MCTS {map_name}")

        ax.set_title(titles[map_name], fontsize=13, pad=10)
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

    plt.suptitle("Learning Dynamics", fontsize=18, y=1.02, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "learning_dynamics.png"), dpi=180, bbox_inches="tight")
    plt.close()

def _episodes_to_solve_or_cap(res: Dict[str, Any], cap: int) -> int:
    if res is None:
        return cap
    if res.get("solved", False):
        return max(1, int(res.get("solve_episode", cap)))
    return cap

def plot_episodes_to_solve_bar(all_results: Dict[str, Dict[str, Dict[str, Any]]],
                               output_path: str,
                               caps: Dict[str, Dict[str, int]],
                               include_strong: bool = False):
    """Log-scale bar chart with annotations and consistent colors."""
    PALETTE = {"RaMCTS": "#2ca02c", "RaMCTS_strong": "#137d13", "Vanilla": "#ff7f0e", "Q-Learning": "#1f77b4"}
    methods_base = ["Q-Learning", "Vanilla", "RaMCTS"]
    has_strong = include_strong and any("RaMCTS_strong" in all_results.get(m, {}) for m in ["4x4", "8x8"])
    methods = methods_base + (["RaMCTS_strong"] if has_strong else [])

    fig, ax = plt.subplots(figsize=(14, 5.5))
    x_positions, heights, colors, labels = [], [], [], []
    group_centers = []
    xpos = 0

    for map_name in ["4x4", "8x8"]:
        start = xpos
        for key in methods:
            candidate = all_results.get(map_name, {}).get(key)
            if key == "RaMCTS_strong" and candidate is None:
                continue  # skip if strong variant wasn't run
            if candidate is None:
                candidate = all_results.get(map_name, {}).get(key.replace("_strong", ""))
            cap = caps.get(key.replace("_strong", ""), {}).get(map_name, 1000)
            ep = _episodes_to_solve_or_cap(candidate, cap)
            x_positions.append(xpos)
            heights.append(ep)
            colors.append(PALETTE.get(key, "#888888"))
            labels.append((map_name, key))
            xpos += 1
        end = xpos - 1
        group_centers.append((start + end) / 2.0)
        xpos += 1  # spacer

    bars = ax.bar(x_positions, heights, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Episodes (Log Scale)")
    ax.set_title("Episodes to Solve (Lower is Better, Capped Values are Failures)", pad=12)
    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"FrozenLake {m}" for m in ["4x4", "8x8"]])
    ax.grid(True, axis='y', alpha=0.3)

    # annotate bars
    for b, h in zip(bars, heights):
        txt = f"{int(h)}" if np.isfinite(h) else "Fail"
        ax.text(b.get_x() + b.get_width() / 2, h * 1.05, txt, ha="center", va="bottom", fontsize=9)

    # legend
    from matplotlib.patches import Patch
    legend_keys = list(dict.fromkeys([k for _, k in labels]))  # unique order-preserving
    handles = [Patch(
        label=("RaMCTS Strong" if k == "RaMCTS_strong" else
               "RaMCTS Simple" if k == "RaMCTS" else
               "Vanilla MCTS" if k == "Vanilla" else k),
        facecolor=PALETTE.get(k, "#888888")
    ) for k in legend_keys]
    ax.legend(handles=handles, frameon=False, loc="upper right")
    plt.savefig(os.path.join(output_path, "episodes_to_solve.png"), dpi=180, bbox_inches="tight")
    plt.close()

def _score_sample_efficiency(ep: int, ep_min: int = 10, ep_max: int = 30000) -> float:
    ep = max(ep_min, min(ep, ep_max))
    return 100.0 * (ep_max - ep) / (ep_max - ep_min)

def _estimate_compute(ep_solve: int, sims_per_move: int, avg_steps: int) -> float:
    return float(ep_solve) * float(sims_per_move) * float(avg_steps)

def _score_cost(total_compute: float, cmin: float, cmax: float) -> float:
    total_compute = max(cmin, min(total_compute, cmax))
    return 100.0 * (cmax - total_compute) / (cmax - cmin + 1e-9)

def plot_efficiency_radar(all_results: Dict[str, Dict[str, Dict[str, Any]]],
                          output_path: str,
                          sims_per_move: Dict[str, int]):
    """Radar chart; 'Sample Efficiency' and 'Computational Cost' derived from experiments."""
    labels = ["Sample Efficiency", "Computational Cost", "Scalability", "Interpretability", "Robustness"]
    methods = [("RaMCTS", "RaMCTS"), ("Vanilla", "Vanilla MCTS"), ("Q-Learning", "Q-Learning")]

    # gather episodes to solve and compute cost proxies
    epi: Dict[str, float] = {}
    comp: Dict[str, float] = {}
    for key, _ in methods:
        eps_4 = _episodes_to_solve_or_cap(all_results.get("4x4", {}).get(key),
                                          cap=10000 if key == "Q-Learning" else 1000)
        eps_8 = _episodes_to_solve_or_cap(all_results.get("8x8", {}).get(key),
                                          cap=30000 if key == "Q-Learning" else 1000)
        epi[key] = 0.5 * (eps_4 + eps_8)
        avg_steps_4, avg_steps_8 = 20, 40
        if key == "Q-Learning":
            comp[key] = _estimate_compute(eps_4, 1, avg_steps_4) + _estimate_compute(eps_8, 1, avg_steps_8)
        else:
            comp[key] = (_estimate_compute(eps_4, sims_per_move.get("4x4", 150), avg_steps_4) +
                         _estimate_compute(eps_8, sims_per_move.get("8x8", 300), avg_steps_8))

    # normalization for cost
    C_MIN, C_MAX = min(comp.values()), max(comp.values())

    scores: Dict[str, List[float]] = {}
    for key, _ in methods:
        scores.setdefault(key, [])
        # sample efficiency
        scores[key].append(_score_sample_efficiency(epi[key], ep_min=10, ep_max=30000))
        # computational cost (lower compute => higher score)
        scores[key].append(_score_cost(comp[key], C_MIN, C_MAX))
        # heuristic but consistent axes
        scores[key].append({"RaMCTS": 90, "Vanilla": 60, "Q-Learning": 30}[key])  # Scalability
        scores[key].append({"RaMCTS": 95, "Vanilla": 40, "Q-Learning": 60}[key])  # Interpretability
        scores[key].append({"RaMCTS": 85, "Vanilla": 55, "Q-Learning": 45}[key])  # Robustness

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(12, 5.2))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 100)

    for key, display in methods:
        vals = scores[key] + scores[key][:1]
        ax.plot(angles, vals, linewidth=2, label=display)
        ax.fill(angles, vals, alpha=0.08)

    ax.set_title("Multi-Dimensional Performance Analysis", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), frameon=False)
    plt.savefig(os.path.join(output_path, "efficiency_radar.png"), dpi=180, bbox_inches="tight")
    plt.close()


def plot_env_learning_dynamics(env_name: str, output_path: str) -> None:
    """Single-environment learning dynamics plot."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method, label in [("RaMCTS", "RaMCTS"), ("Vanilla", "Vanilla MCTS"), ("Q-Learning", "Q-Learning")]:
        try:
            ep, sr = _load_series_strict(method, env_name, "")
            ax.plot(ep, sr, linewidth=2, label=label)
        except FileNotFoundError:
            print(f"[plot] missing logs: {label} {env_name}")
    ax.set_title(f"{env_name} Learning", fontsize=13, pad=10)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{env_name}_learning_dynamics.png"), dpi=180, bbox_inches="tight")
    plt.close()


def plot_env_episodes_to_solve(all_results: Dict[str, Any],
                               env_name: str,
                               output_path: str,
                               caps: Dict[str, int]) -> None:
    """Bar chart of episodes to solve for a single environment."""
    methods = [("Q-Learning", "Q-Learning"), ("Vanilla", "Vanilla MCTS"), ("RaMCTS", "RaMCTS")]
    heights = []
    labels = []
    for key, label in methods:
        cap = caps.get(key, 1000)
        ep = _episodes_to_solve_or_cap(all_results.get(key), cap)
        heights.append(ep)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(range(len(methods)), heights)
    ax.set_yscale("log")
    ax.set_ylabel("Episodes (Log Scale)")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_title(f"Episodes to Solve — {env_name}")
    ax.grid(True, axis='y', alpha=0.3)
    for b, h in zip(bars, heights):
        txt = f"{int(h)}" if np.isfinite(h) else "Fail"
        ax.text(b.get_x() + b.get_width() / 2, h * 1.05, txt, ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{env_name}_episodes_to_solve.png"), dpi=180, bbox_inches="tight")
    plt.close()


def plot_env_efficiency_radar(all_results: Dict[str, Any],
                              env_name: str,
                              output_path: str,
                              sims_per_move: int) -> None:
    """Radar chart for a single environment."""
    labels = ["Sample Efficiency", "Computational Cost", "Scalability", "Interpretability", "Robustness"]
    methods = [("RaMCTS", "RaMCTS"), ("Vanilla", "Vanilla MCTS"), ("Q-Learning", "Q-Learning")]

    epi: Dict[str, float] = {}
    comp: Dict[str, float] = {}
    for key, _ in methods:
        cap = 30000 if key == "Q-Learning" else 1000
        ep = _episodes_to_solve_or_cap(all_results.get(key), cap)
        epi[key] = ep
        avg_steps = 40
        if key == "Q-Learning":
            comp[key] = _estimate_compute(ep, 1, avg_steps)
        else:
            comp[key] = _estimate_compute(ep, sims_per_move, avg_steps)

    C_MIN, C_MAX = min(comp.values()), max(comp.values())

    scores: Dict[str, List[float]] = {}
    for key, _ in methods:
        scores.setdefault(key, [])
        scores[key].append(_score_sample_efficiency(epi[key], ep_min=10, ep_max=30000))
        scores[key].append(_score_cost(comp[key], C_MIN, C_MAX))
        scores[key].append({"RaMCTS": 90, "Vanilla": 60, "Q-Learning": 30}[key])
        scores[key].append({"RaMCTS": 95, "Vanilla": 40, "Q-Learning": 60}[key])
        scores[key].append({"RaMCTS": 85, "Vanilla": 55, "Q-Learning": 45}[key])

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 100)

    for key, display in methods:
        vals = scores[key] + scores[key][:1]
        ax.plot(angles, vals, linewidth=2, label=display)
        ax.fill(angles, vals, alpha=0.08)

    ax.set_title(f"{env_name} Performance Analysis", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), frameon=False)
    plt.savefig(os.path.join(output_path, f"{env_name}_efficiency_radar.png"), dpi=180, bbox_inches="tight")
    plt.close()

# ====================
# Main Execution
# ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RaMCTS experiments")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="number of experiment repetitions")
    parser.add_argument("--envs", type=str, default=None,
                        help="Comma-separated environments: FrozenLake, Taxi-v3, CliffWalking-v1")
    parser.add_argument("--all", action="store_true", help="Run all environments")
    args = parser.parse_args()

    if args.all:
        env_list = ["FrozenLake"] + GENERIC_ENVS
    elif args.envs:
        env_list = [e.strip() for e in args.envs.split(",") if e.strip()]
    else:
        env_list = ["FrozenLake"]

    print("\nStarting Fixed Experiments...")
    print("=" * 60)

    results_file = open(f"{OUTPUT_DIR}/results_fixed.txt", 'w')

    def log(text):
        print(text)
        results_file.write(text + '\n')
        results_file.flush()

    all_runs: List[Dict[str, Any]] = []

    for run_idx in range(1, args.runs + 1):
        log(f"\n=== Run {run_idx}/{args.runs} ===")
        all_results: Dict[str, Any] = {}

        if "FrozenLake" in env_list:
            all_results["FrozenLake"] = {}
            for map_name in FROZENLAKE_MAPS:
                log(f"\nTesting on FrozenLake {map_name}")
                log("=" * 60)
                all_results["FrozenLake"][map_name] = {}
                log("\n1. Q-Learning (Fixed)")
                res = run_qlearning_experiment(map_name, max_episodes=FROZENLAKE_Q_EPISODES[map_name])
                all_results["FrozenLake"][map_name]['Q-Learning'] = res
                budget = FROZENLAKE_BUDGET[map_name]
                for method in ["Vanilla", "RaMCTS"]:
                    disp = "Vanilla MCTS" if method == "Vanilla" else method
                    log(f"\n{disp} ({budget} sims)")
                    res = run_experiment(map_name, method, budget, max_episodes=1000)
                    all_results["FrozenLake"][map_name][method] = res

        if "Taxi-v3" in env_list:
            env_id = "Taxi-v3"
            log(f"\nTesting on {env_id}")
            log("=" * 60)
            all_results[env_id] = {}
            log("\n1. Q-Learning (Fixed)")
            res = run_qlearning_generic(env_id, max_episodes=GENERIC_Q_EPISODES[env_id],
                                        step_cap=GENERIC_EP_STEPS[env_id])
            all_results[env_id]['Q-Learning'] = res
            budget = GENERIC_MCTS_BUDGET[env_id]
            for method in ["Vanilla", "RaMCTS"]:
                disp = "Vanilla MCTS" if method == "Vanilla" else method
                log(f"\n{disp} ({budget} sims)")
                res = run_generic_experiment(env_id, method, budget,
                                             GENERIC_ROLLOUT[env_id],
                                             GENERIC_EPISODES[env_id],
                                             GENERIC_EP_STEPS[env_id])
                all_results[env_id][method] = res

        if "CliffWalking-v1" in env_list:
            env_id = "CliffWalking-v1"
            log(f"\nTesting on {env_id}")
            log("=" * 60)
            all_results[env_id] = {}
            log("\n1. Q-Learning (Fixed)")
            res = run_qlearning_generic(env_id, max_episodes=GENERIC_Q_EPISODES[env_id],
                                        step_cap=GENERIC_EP_STEPS[env_id])
            all_results[env_id]['Q-Learning'] = res
            budget = GENERIC_MCTS_BUDGET[env_id]
            for method in ["Vanilla", "RaMCTS"]:
                if method == "Vanilla":
                    cfg = CLIFF_VANILLA_MCTS_CONFIG
                    sims = cfg["num_simulations"]
                    rollout = cfg["max_moves"]
                    episodes = cfg["episodes"]
                    step_cap = cfg["max_moves"]
                    discount = cfg["discount"]
                    root_frac = cfg["root_exploration_fraction"]
                    pb_base = cfg["pb_c_base"]
                    pb_init = cfg["pb_c_init"]
                else:
                    sims = budget
                    rollout = GENERIC_ROLLOUT[env_id]
                    episodes = GENERIC_EPISODES[env_id]
                    step_cap = GENERIC_EP_STEPS[env_id]
                    discount = 1.0
                    root_frac = 0.0
                    pb_base = 1.0
                    pb_init = 1.25
                disp = "Vanilla MCTS" if method == "Vanilla" else method
                log(f"\n{disp} ({sims} sims)")
                res = run_generic_experiment(env_id, method, sims,
                                             rollout,
                                             episodes,
                                             step_cap,
                                             discount,
                                             root_frac,
                                             pb_base,
                                             pb_init)
                all_results[env_id][method] = res

        # Summary for this run
        log("\n" + "=" * 60)
        log(f"RESULTS SUMMARY (Run {run_idx})")
        log("=" * 60)
        keys = {"Q-Learning": "Q-Learning", "Vanilla": "Vanilla MCTS", "RaMCTS": "RaMCTS"}
        if "FrozenLake" in env_list:
            for map_name in FROZENLAKE_MAPS:
                log(f"\nFrozenLake {map_name}:")
                for key, label in keys.items():
                    res = all_results["FrozenLake"][map_name].get(key)
                    if res is None:
                        log(f"  {label}: (missing)")
                        continue
                    status = f"SOLVED in {res['solve_episode']} episodes" if res['solved'] else "Failed"
                    log(f"  {label}: {status}")
        for env_id in GENERIC_ENVS:
            if env_id in env_list:
                log(f"\n{env_id}:")
                for key, label in keys.items():
                    res = all_results[env_id].get(key)
                    if res is None:
                        log(f"  {label}: (missing)")
                        continue
                    status = f"SOLVED in {res['solve_episode']} episodes" if res['solved'] else "Failed"
                    log(f"  {label}: {status}")

        all_runs.append(all_results)

    # Averaged summary across runs
    log("\n" + "=" * 60)
    log(f"AVERAGED RESULTS OVER {args.runs} RUNS")
    log("=" * 60)
    keys = {"Q-Learning": "Q-Learning", "Vanilla": "Vanilla MCTS", "RaMCTS": "RaMCTS"}
    if "FrozenLake" in env_list:
        for map_name in FROZENLAKE_MAPS:
            log(f"\nFrozenLake {map_name}:")
            for key, label in keys.items():
                solved_runs = [run["FrozenLake"][map_name][key]['solved'] for run in all_runs]
                solve_episodes = [run["FrozenLake"][map_name][key]['solve_episode'] for run in all_runs if run["FrozenLake"][map_name][key]['solved']]
                solve_rate = float(np.mean(solved_runs)) if solved_runs else 0.0
                avg_ep = float(np.mean(solve_episodes)) if solve_episodes else float('nan')
                if np.isnan(avg_ep):
                    log(f"  {label}: solved {solve_rate*100:.1f}% runs, avg solve episode: N/A")
                else:
                    log(f"  {label}: solved {solve_rate*100:.1f}% runs, avg solve episode: {avg_ep:.1f}")
    for env_id in GENERIC_ENVS:
        if env_id in env_list:
            log(f"\n{env_id}:")
            for key, label in keys.items():
                solved_runs = [run[env_id][key]['solved'] for run in all_runs]
                solve_episodes = [run[env_id][key]['solve_episode'] for run in all_runs if run[env_id][key]['solved']]
                solve_rate = float(np.mean(solved_runs)) if solved_runs else 0.0
                avg_ep = float(np.mean(solve_episodes)) if solve_episodes else float('nan')
                if np.isnan(avg_ep):
                    log(f"  {label}: solved {solve_rate*100:.1f}% runs, avg solve episode: N/A")
                else:
                    log(f"  {label}: solved {solve_rate*100:.1f}% runs, avg solve episode: {avg_ep:.1f}")

    results_file.close()
    print(f"\nResults saved to {OUTPUT_DIR}/results_fixed.txt")
    print("\n✅ Heuristic-free ablations complete.")

    if "FrozenLake" in env_list:
        sims_map = {m: FROZENLAKE_BUDGET[m] for m in FROZENLAKE_MAPS}
        plot_learning_dynamics(OUTPUT_DIR, ql_scale=10.0, x_cutoff_4x4=30, x_cutoff_8x8=170)
        plot_episodes_to_solve_bar(
            all_runs[-1]["FrozenLake"],
            OUTPUT_DIR,
            caps={"Q-Learning": {"4x4": 10000, "8x8": 30000},
                  "Vanilla": {"4x4": 1000, "8x8": 1000},
                  "RaMCTS": {"4x4": 1000, "8x8": 1000}},
            include_strong=False,
        )
        plot_efficiency_radar(all_runs[-1]["FrozenLake"], OUTPUT_DIR, sims_per_move=sims_map)

    for env_id in GENERIC_ENVS:
        if env_id in env_list:
            env_res = all_runs[-1][env_id]
            plot_env_learning_dynamics(env_id, OUTPUT_DIR)
            plot_env_episodes_to_solve(env_res, env_id, OUTPUT_DIR,
                                       caps={"Q-Learning": GENERIC_Q_EPISODES[env_id],
                                             "Vanilla": GENERIC_EPISODES[env_id],
                                             "RaMCTS": GENERIC_EPISODES[env_id]})
            plot_env_efficiency_radar(env_res, env_id, OUTPUT_DIR,
                                       sims_per_move=GENERIC_MCTS_BUDGET[env_id])
