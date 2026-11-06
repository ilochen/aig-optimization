import os
import subprocess
import tempfile
import shutil
import random
import math
from helper_functions import strip_ansi_escape_sequences, parse_abc_stats_line, detect_format
from copy import deepcopy
from tqdm import tqdm, trange
import uuid

TEST_BENCH = "./testbenches/2025_IWLS_Contest_Benchmarks_020425/ex100.truth"

import itertools
import tempfile
import os

class ABCEnv:
    def __init__(self, abc_path="./abc", truth_file=TEST_BENCH, alpha=1.0, beta=0.3):
        self.layer = 0
        self._aig_copy_index = 1
        self.abc_path = abc_path
        self.benchmark_path = truth_file
        self.benchmark_type = detect_format(truth_file)

        # === Dynamic action space generation ===
    
        self.action_space = self._randomize_action_space()

        # === Workspace setup ===
        self._temp_dir = tempfile.TemporaryDirectory()
        self._work_dir = self._temp_dir.name
        self._current_aig = os.path.join(self._work_dir, "current.aig")

        # Reward weights
        self.alpha = alpha
        self.beta = beta

        # Tracking
        self.prev_obs = None
        self.initial_obs = None

        # State-action LUT
        self.optimization_lut = {}
        self.lut_threshold = 0.1

        # Initialize environment
        self.reset()
        
    import random

    def _randomize_action_space(self):
        # Fixed command categories
        actions = []

        # 1. Simple static actions
        actions.extend(["rewrite", "rewrite -z", "balance", "refactor", "refactor -z", "resub -z"])

        # 2. Randomized resub variant
        k = random.choice([6, 8, 10, 12])
        if random.random() < 0.5:
            n = random.choice([2])
            actions.append(f"resub -K {k} -N {n}")
        else:
            actions.append(f"resub -K {k}")

        # 3. Randomized ABC9 macro command
        dch = random.choice(["&dch", "&dch -f"])
        k_if = random.choice([3, 4, 5])
        macro_cmd = (
            f"{dch}; "
            f"&if -a -K {k_if}; "
            f"&mfs -e -W 20 -L 20; "
            f"&fx; &st; &put; balance"
        )
        actions.append(macro_cmd)

        return actions



    def reset(self):
        if self.benchmark_type == "truth":
            commands = [
                f"read_truth -xf {self.benchmark_path}",
                "collapse",
                "strash",
                f"write {self._current_aig}"
            ]
        elif self.benchmark_type == "aig":
            commands = [
                f"read {self.benchmark_path}",
                f"write {self._current_aig}"
            ]
        else:
            raise RuntimeError("Unknown benchmark type.")

        self._run_abc(commands)
        obs = self._get_observation()
        self.prev_obs = obs
        if self.initial_obs is None:
            self.initial_obs = obs
        return obs

    def calculate_reward(self, current_obs):
        """More comprehensive reward calculation"""
        if self.prev_obs is None:
            return 0.0
        
        # Area estimation with interconnect consideration
        current_area = current_obs['ands'] * (1 + math.log2(current_obs['ands'] / current_obs['levels']))
        prev_area = self.prev_obs['ands'] * (1 + math.log2(self.prev_obs['ands'] / self.prev_obs['levels']))
        
        # Delay estimation with fanout effects
        current_delay = current_obs['levels'] * (1 + 0.1 * math.log2(current_obs['ands'] / current_obs['levels']))
        prev_delay = self.prev_obs['levels'] * (1 + 0.1 * math.log2(self.prev_obs['ands'] / self.prev_obs['levels']))
        
        # Relative improvements
        area_improvement = (prev_area - current_area) #/ prev_area
        delay_improvement = (prev_delay - current_delay) / prev_delay
        
        # Combined reward with technology-specific weights
        # reward = (self.alpha * area_improvement + 
        #          self.beta * delay_improvement)
        reward = self.alpha * area_improvement 
        
        return reward

    def step(self, action_index):
        command = self.action_space[action_index]
        temp_aig = os.path.join(self._work_dir, "temp.aig")

        commands = [
            f"read {self._current_aig}",
            f"&r {self._current_aig}",
            command,
            f"write {temp_aig}"
        ]
        self._run_abc(commands)
        shutil.move(temp_aig, self._current_aig)
        print(f"[DEBUG] Step applied: {command}, updated AIG: {self._current_aig}")
        current_obs = self._get_observation()
        reward = self.calculate_reward(current_obs)
        done = False
        
        # Update previous observation
        self.prev_obs = current_obs
        
        # Update LUT with successful patterns
        if reward > 0:
            state_hash = self.get_state_hash(self.prev_obs)
            improvement = (self.prev_obs['ands'] - current_obs['ands']) / self.prev_obs['ands']
            self.update_lut(state_hash, action_index, improvement)
        
        return current_obs, reward, done, {}

    def _get_observation(self):
        commands = [
            f"read {self._current_aig}",
            "print_stats"
        ]
        output = self._run_abc(commands)
        stats_line = [l for l in output.splitlines() if "i/o" in l][-1]
        stats_line = strip_ansi_escape_sequences(stats_line)
        return parse_abc_stats_line(stats_line)

    def _run_abc(self, commands):
        joined = "\n".join(commands) + "\nquit\n"
        result = subprocess.run(
            [self.abc_path],
            input=joined.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return result.stdout.decode()

    def close(self):
        #shutil.rmtree(self._work_dir)
        pass

    def render(self):
        obs = self._get_observation()
        print(f"ANDs: {obs['ands']} | Levels: {obs['levels']}")

    def get_state_hash(self, obs):
        """Create a hashable state representation"""
        return (obs['ands'], obs['levels'])
        
    def update_lut(self, state_hash, action, improvement):
        """Store successful optimization patterns"""
        if improvement > self.lut_threshold:
            if state_hash not in self.optimization_lut:
                self.optimization_lut[state_hash] = []
            self.optimization_lut[state_hash].append((action, improvement))

    def analyze_circuit_structure(self, obs):
        """Analyze circuit structure for LUT matching"""
        # Could include:
        # - AND/level ratio
        # - Local structure patterns
        # - Critical path characteristics
        structure_features = {
            'and_level_ratio': obs['ands'] / obs['levels'],
            'size_category': obs['ands'] // 100,  # Group by size ranges
            'depth_category': obs['levels'] // 10
        }
        return structure_features
    
    def __deepcopy__(self, memo):
        # Create a new environment with the same init parameters
        copied = ABCEnv(
            abc_path=self.abc_path,
            truth_file=self.benchmark_path,
            alpha=self.alpha,
            beta=self.beta
        )
        
        #copy the action space
        copied.action_space = deepcopy(self.action_space, memo)

        # Copy observations and LUT if needed
        copied.prev_obs = deepcopy(self.prev_obs, memo)
        copied.initial_obs = deepcopy(self.initial_obs, memo)
        copied.optimization_lut = deepcopy(self.optimization_lut, memo)

       # Use sequential numbering for temp AIG files
        self._aig_copy_index += 1
        copied._aig_copy_index = self._aig_copy_index

        new_aig_path = os.path.join(copied._work_dir, f"copied_{copied._aig_copy_index}.aig")
        shutil.copy(self._current_aig, new_aig_path)
        copied._current_aig = new_aig_path

        print(f"[DEBUG] Deepcopied AIG to: {new_aig_path}")

        return copied
    
    def export_final_aig(self, output_path):
        """
        Copy the current AIG file to a permanent location.
        """
        shutil.copy(self._current_aig, output_path)
        print(f"[INFO] Final AIG saved to: {output_path}")
        
    def reroll_action_space(self, complex = False):
        """
        Regenerate the randomized variants of resub and ABC9 macro command.
        """
        actions = []

        # Static base commands
        actions.extend(["rewrite", "rewrite -z", "balance", "refactor", "refactor -z", "resub -z"])

        # Reroll resub command
        if complex:
            # append multiple resub commands with different K and N values
            for k in [6, 8, 10, 12]:
                if random.random() < 0.5:
                    n = random.choice([2])
                    actions.append(f"resub -K {k} -N {n}")
                else:
                    actions.append(f"resub -K {k}")
        else:
            k = random.choice([6, 8, 10, 12])
            if random.random() < 0.5:
                n = random.choice([2])
                actions.append(f"resub -K {k} -N {n}")
            else:
                actions.append(f"resub -K {k}")

        # Reroll ABC9 macro command
        dch = random.choice(["&dch", "&dch -f"])
        k_if = random.choice([3, 4, 5])
        macro_cmd = (
            f"{dch}; "
            f"&if -a -K {k_if}; "
            f"&mfs -e -W 20 -L 20; "
            f"&fx; &st; &put; balance"
        )
        actions.append(macro_cmd)

        self.action_space = actions
        print("[INFO] Action space re-rolled:")
        for i, act in enumerate(self.action_space):
            print(f"  [{i}] {act}")


class MCTSNode:
    def __init__(self, state=None, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0
        self.untried_actions = []

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def best_child(self, c_param=1.414):
        choices = [(child.value / child.visits + c_param * math.sqrt(2 * math.log(self.visits) / child.visits), action, child)
                  for action, child in self.children.items()]
        return max(choices, key=lambda x: x[0])[1:]

    def expand(self, action, state):
        child = MCTSNode(state=state, parent=self, action=action)
        self.children[action] = child
        if self.untried_actions is not None:
            self.untried_actions.remove(action)
        return child

class MacroMCTS:
    def __init__(self, env, iterations=500, rollout_depth=3, exploration_weight=1.414, sample_k=10):
        self.env = env
        self.iterations = iterations
        self.rollout_depth = rollout_depth
        self.exploration_weight = exploration_weight
        self.sample_k = sample_k
        self.lut_use_probability = 0.7  # Probability to use LUT during rollout

    def search(self, initial_state):
        root = MCTSNode(state=initial_state)
        root.untried_actions = list(range(len(self.env.action_space)))
        #root.untried_actions = random.sample(range(len(self.env.action_space)), self.sample_k)
        for _ in range(self.iterations):
            layer = 0
            node = root
            env_state = deepcopy(self.env)

            # Selection
            while node.is_fully_expanded() and node.children:
                layer += 1
                action, node = node.best_child(self.exploration_weight)
                _, _, _, _ = env_state.step(action)

            # Expansion
            rollout_reward = 0
            if not node.is_fully_expanded():
                print(f"[DEBUG] Expanding node at layer {layer} with untried actions: {node.untried_actions}")
                env_state.render()
                action = random.choice(node.untried_actions)
                obs, reward, done, _ = env_state.step(action)
                rollout_reward += reward
                new_node = node.expand(action, obs)
                #new_node.untried_actions = random.sample(range(len(self.env.action_space)), self.sample_k)
                new_node.untried_actions = list(range(len(self.env.action_space)))
                node = new_node

            # Rollout
            # rollout_reward = 0
            # for _ in range(self.rollout_depth):
            #     action = random.randint(0, len(self.env.action_space) - 1)
            #     _, reward, done, _ = env_state.step(action)
            #     rollout_reward += reward

            # Backpropagation
            while node:
                node.visits += 1
                node.value += rollout_reward
                node = node.parent

        #return root.best_child(c_param=0)[0]
        actions = []
        node = root
        for _ in range(10):
            if not node.children:
                break
            _, action, best_node = max(
                [(child.value / child.visits, a, child) for a, child in node.children.items()],
                key=lambda x: x[0]
            )
            actions.append(action)
            node = best_node
        return actions

    def rollout_policy(self, env_state):
        """Smart rollout using LUT when available"""
        total_reward = 0
        
        for _ in range(self.rollout_depth):
            current_state = env_state._get_observation()
            state_hash = env_state.get_state_hash(current_state)
            
            if state_hash in env_state.optimization_lut and random.random() < self.lut_use_probability:
                lut_actions = env_state.optimization_lut[state_hash]
                action, _ = max(lut_actions, key=lambda x: x[1])
            else:
                action = random.randint(0, len(self.env.action_space) - 1)
                
            obs, reward, done, _ = env_state.step(action)
            total_reward += reward

if __name__ == "__main__":
    # Initialize environment with reward weights
    env = ABCEnv(
        abc_path="./abc/abc",
        truth_file=TEST_BENCH,
        alpha=1.0,    # Weight for node count reduction
        beta=0.3      # Weight for logic depth reduction
    )

    # Initialize MCTS with appropriate parameters
    mcts = MacroMCTS(
        env, 
        iterations=100,
        rollout_depth=0,
        exploration_weight=1.414
    )
    
    print("=== Starting MCTS optimization ===")
    obs = env.reset()
    initial_ands = obs['ands']
    initial_levels = obs['levels']
    best_ands = initial_ands
    best_levels = initial_levels
    steps_without_improvement = 0
    
    # for step in tqdm(range(60), desc="Optimizing MCTS"):
    #     action = mcts.search(obs)
    #     print(f"\nStep {step+1}: Action = {env.action_space[action]}")
    #     obs, reward, done, _ = env.step(action)
        
    #     print(f"ANDs: {obs['ands']} (Initial: {initial_ands})")
    #     print(f"Levels: {obs['levels']} (Initial: {initial_levels})")
    #     print(f"Step Reward: {reward:.2f}")
        
    #     # Track improvements in both metrics
    #     if obs['ands'] < best_ands or obs['levels'] < best_levels:
    #         and_improvement = ((best_ands - obs['ands']) / best_ands) * 100
    #         level_improvement = ((best_levels - obs['levels']) / best_levels) * 100
    #         print(f"Improvements - ANDs: {and_improvement:.2f}% | Levels: {level_improvement:.2f}%")
    #         best_ands = min(best_ands, obs['ands'])
    #         best_levels = min(best_levels, obs['levels'])
    complex = False
    for step in tqdm(range(60), desc="Optimizing MCTS"):
        action_plan = mcts.search(obs)  # return top 5 actions
        total_reward = 0
        for idx, action in enumerate(action_plan):
            print(f"\nStep {step+1}.{idx+1}: Action = {env.action_space[action]}")
            obs, reward, done, _ = env.step(action)

            print(f"ANDs: {obs['ands']} (Initial: {initial_ands})")
            print(f"Levels: {obs['levels']} (Initial: {initial_levels})")
            print(f"Step Reward: {reward:.2f}")

            total_reward += reward

            # Track improvements in both metrics
            if obs['ands'] < best_ands or obs['levels'] < best_levels:
                and_improvement = ((best_ands - obs['ands']) / best_ands) * 100
                level_improvement = ((best_levels - obs['levels']) / best_levels) * 100
                print(f"Improvements - ANDs: {and_improvement:.2f}% | Levels: {level_improvement:.2f}%")
                best_ands = min(best_ands, obs['ands'])
                best_levels = min(best_levels, obs['levels'])
        #reroll the action space for each step
        if total_reward < 100:
            #make an uphill move (last index of action space)
            index = len(env.action_space) - 1
            env.step(index)
        if total_reward < 200 and mcts.iterations < 300:
            mcts.iterations += 50   
        if total_reward < 500:
            complex = True
            # mcts.iterations += 50
        if total_reward > 2000 and mcts.iterations > 100:
            mcts.iterations -= 50
        env.reroll_action_space(complex)

    env.export_final_aig("final_output.aig")

    
    print(f"\nOptimization complete:")
    print(f"Initial ANDs: {initial_ands} -> Final: {obs['ands']} ({((initial_ands - obs['ands']) / initial_ands) * 100:.2f}%)")
    print(f"Initial Levels: {initial_levels} -> Final: {obs['levels']} ({((initial_levels - obs['levels']) / initial_levels) * 100:.2f}%)")

    env.close()