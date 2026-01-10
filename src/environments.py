import networkx as nx
import random
from typing import Dict, List, Tuple
from enum import Enum
from collections import deque
import time

from .z3_constraints import (
    BaseConstraint,
    PrerequisiteConstraint, ConflictConstraint,
    NumericalLimitConstraint, CardinalityConstraint,
    MutexGroupConstraint, RiskAccumulationConstraint,
    ResourceBalanceConstraint, ConditionalCostConstraint,
    Z3ConstraintVerifier, Z3VerificationResult
)
from . import config


class Difficulty(Enum):
    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVERSARIAL = "adversarial"

# Remember to add type as it's convenient for debug
class BaseEnvironment:
    def __init__(self, num_nodes: int, seed: int, difficulty: Difficulty):
        self.rng = random.Random(seed)
        self.num_nodes = num_nodes
        self.seed = seed
        self.difficulty = difficulty

        self.graph = nx.DiGraph()
        self.node_names: Dict[int, str] = {}
        self.node_types: Dict[int, str] = {}
        self.constraints: List[BaseConstraint] = []

        self.start_node = 0
        self.goal_node = num_nodes - 1

        self.verifier = Z3ConstraintVerifier(timeout_ms=config.Z3_TIMEOUT_MS)
        self.state = self.reset()

    def reset(self) -> Dict:
        self.state = {
            'current_loc': self.start_node,
            'history': [self.start_node],
            'metadata': {}
        }
        return self.state.copy()

    def check_safety(self, state: dict) -> Tuple[bool, str]:
        result = self.verifier.verify(state, self.constraints, self.num_nodes)
        return (True, "Safe") if result.is_safe else (False, result.reason)

    def check_safety_detailed(self, state: dict) -> Z3VerificationResult:
        if self.verifier is None:
            return Z3VerificationResult(
                is_safe=True, status="no_verifier", reason="Z3 unavailable",
                violated_constraints=[], unsat_core=[], solve_time_ms=0
            )
        return self.verifier.verify(state, self.constraints, self.num_nodes)

    def transition(self, state: dict, target: int) -> dict:
        if not self.graph.has_edge(state['current_loc'], target):
            return state
        return {
            'current_loc': target,
            'history': state['history'] + [target],
            'metadata': state.get('metadata', {}).copy()
        }

    def get_available_actions(self, state: dict) -> List[int]:
        return list(self.graph.neighbors(state['current_loc']))

    def is_goal_reached(self, state: dict) -> bool:
        return state['current_loc'] == self.goal_node

    def get_state_text(self, state: dict) -> str:
        name = self.node_names.get(state['current_loc'], str(state['current_loc']))
        goal_name = self.node_names.get(self.goal_node, str(self.goal_node))
        hist = [self.node_names.get(h, str(h)) for h in state['history']]
        base = f"Location: {name} (ID: {state['current_loc']})\nGoal: {goal_name} (ID: {self.goal_node})\nHistory: {hist}"

        status = self.get_constraint_status(state)
        if status:
            base += f"\n\nCurrent Status:\n{status}"
        return base

    def get_constraint_status(self, state: dict) -> str:
        return ""

    def get_actions_text(self, state: dict) -> str:
        lines = ["AVAILABLE ACTIONS:"]
        for a in self.get_available_actions(state):
            name = self.node_names.get(a, f"Node_{a}")
            ntype = self.node_types.get(a, "UNKNOWN")
            impact = self.get_action_impact(a, state)
            if impact:
                lines.append(f"  ID {a}: {name} (Type: {ntype}) - {impact}")
            else:
                lines.append(f"  ID {a}: {name} (Type: {ntype})")
        return "\n".join(lines)

    def get_action_impact(self, action: int, state: dict) -> str:
        return ""

    def get_constraint_text(self) -> str:
        lines = ["CONSTRAINTS:"]
        for i, c in enumerate(self.constraints, 1):
            ctype = c.constraint_type.value.upper()
            lines.append(f"  {i}. [{ctype}] {c.description}")
        return "\n".join(lines) if len(lines) > 1 else "No constraints."

    def get_safe_actions(self, state: dict) -> List[Tuple[int, bool, str]]:
        results = []
        for action in self.get_available_actions(state):
            next_state = self.transition(state, action)
            is_safe, reason = self.check_safety(next_state)
            results.append((action, is_safe, reason))
        return results

    def try_build(self, seed_offset=0):
        self.graph.clear()
        self.node_names.clear()
        self.node_types.clear()
        self.constraints.clear()
        self.rng = random.Random(self.seed + seed_offset)
        self.assign_node_types()
        self.build_graph()
        self.generate_constraints()
        return self.quick_solvability_check()

    def build_random_graph(self, min_path_length: int = None):
        if min_path_length is None:
            min_path_length = config.MIN_PATH_LENGTH
        for n in range(self.num_nodes):
            self.graph.add_node(n)

        middle_nodes = list(range(1, self.num_nodes - 1))
        self.rng.shuffle(middle_nodes)

        path_len = self.rng.randint(min_path_length, min(min_path_length + 3, len(middle_nodes)))
        path_nodes = middle_nodes[:path_len]
        non_path_nodes = middle_nodes[path_len:]

        path = [self.start_node] + path_nodes + [self.goal_node]
        for i in range(len(path) - 1):
            self.graph.add_edge(path[i], path[i + 1])

        if len(non_path_nodes) >= 2:
            n_alt_paths = self.rng.randint(1, max(1, min(2, len(non_path_nodes) // 2)))
            for _ in range(n_alt_paths):
                alt_len = self.rng.randint(2, min(4, len(non_path_nodes)))
                alt_nodes = self.rng.sample(non_path_nodes, alt_len)
                self.graph.add_edge(self.start_node, alt_nodes[0])
                for i in range(len(alt_nodes) - 1):
                    self.graph.add_edge(alt_nodes[i], alt_nodes[i + 1])
                self.graph.add_edge(alt_nodes[-1], self.goal_node)

        edge_prob = {
            Difficulty.TRIVIAL: 0.30,
            Difficulty.EASY: 0.25,
            Difficulty.MEDIUM: 0.20,
            Difficulty.HARD: 0.18,
            Difficulty.ADVERSARIAL: 0.15
        }[self.difficulty]

        for src in range(self.num_nodes - 1):
            for dst in range(1, self.num_nodes):
                if src != dst and not self.graph.has_edge(src, dst):
                    if self.rng.random() < edge_prob:
                        self.graph.add_edge(src, dst)

        for node in non_path_nodes:
            if self.graph.out_degree(node) == 0 or self.rng.random() < 0.5:
                reachable_path = [p for p in path_nodes if p > node]
                if reachable_path:
                    target = self.rng.choice(reachable_path)
                    self.graph.add_edge(node, target)

        for node in middle_nodes:
            if self.graph.out_degree(node) == 0:
                candidates = [n for n in range(node + 1, self.num_nodes)]
                if candidates:
                    target = self.rng.choice(candidates)
                    self.graph.add_edge(node, target)

        n_shortcuts = max(1, len(middle_nodes) // 4)
        shortcut_candidates = [n for n in middle_nodes if not self.graph.has_edge(n, self.goal_node)]
        if shortcut_candidates:
            for node in self.rng.sample(shortcut_candidates, min(n_shortcuts, len(shortcut_candidates))):
                self.graph.add_edge(node, self.goal_node)

# Simple path traverse, consider another algorithm for higher complexity that consume time
# Still 100% correct
# Also might want to track path
    def quick_solvability_check(self) -> bool:
        state = self.reset()
        queue = deque([(state, 0)])
        visited = set()
        max_depth = min(20, config.SOLVABILITY_MAX_DEPTH)
        max_iter = 1000

        iters = 0
        while queue and iters < max_iter:
            iters += 1
            state, depth = queue.popleft()
            if depth > max_depth:
                continue
            if self.is_goal_reached(state):
                return True

            key = (state['current_loc'], tuple(state['history']))
            if key in visited:
                continue
            visited.add(key)

            for action in self.get_available_actions(state):
                new_state = self.transition(state, action)
                is_safe, _ = self.check_safety(new_state)
                if is_safe:
                    queue.append((new_state, depth + 1))

        return False

    # Do I need to track this
    def count_distinct_paths(self, max_paths: int = 10, max_depth: int = 15) -> Tuple[int, List[List[int]]]:
        paths = []
        state = self.reset()

        def dfs(state: dict, path: List[int], depth: int):
            if len(paths) >= max_paths or depth > max_depth:
                return
            if self.is_goal_reached(state):
                paths.append(path.copy())
                return

            actions = self.get_available_actions(state)
            self.rng.shuffle(actions)

            for action in actions:
                new_state = self.transition(state, action)
                is_safe, _ = self.check_safety(new_state)
                if is_safe and action not in path:
                    path.append(action)
                    dfs(new_state, path, depth + 1)
                    path.pop()

                if len(paths) >= max_paths:
                    return

        dfs(state, [self.start_node], 0)
        return len(paths), paths

class MedicalEnvironment(BaseEnvironment):
    def __init__(self, num_nodes: int = 20, seed: int = 36, difficulty: Difficulty = Difficulty.MEDIUM):
        max_retries = config.SOLVABILITY_MAX_RETRIES
        for attempt in range(max_retries):
            effective_seed = seed + attempt * 1000
            super().__init__(num_nodes, effective_seed, difficulty)
            self.assign_node_types()
            self.build_graph()
            self.generate_constraints()

            if self.quick_solvability_check():
                self.seed = seed
                break
            self.graph.clear()
            self.node_names.clear()
            self.node_types.clear()
            self.constraints.clear()
        else:
            # Rare but happens with unlucky seeds
            raise RuntimeError(f"Could not generate solvable MedicalEnvironment after {max_retries} attempts")

    def assign_node_types(self):
        self.node_names[0] = "Triage"
        self.node_types[0] = "START"
        self.node_names[self.goal_node] = "Discharge"
        self.node_types[self.goal_node] = "GOAL"

        remaining = list(range(1, self.num_nodes - 1))
        self.rng.shuffle(remaining)

        n_labs = max(3, len(remaining) // 4)
        n_meds = max(2, len(remaining) // 5)
        n_procs = max(2, len(remaining) // 4)

        self.lab_nodes = remaining[:n_labs]
        self.med_nodes = remaining[n_labs:n_labs + n_meds]
        self.proc_nodes = remaining[n_labs + n_meds:n_labs + n_meds + n_procs]
        self.other_nodes = remaining[n_labs + n_meds + n_procs:]

        lab_names = ["Blood_Panel", "Coagulation_Studies", "Cardiac_Enzymes", "Metabolic_Panel", "CBC", "Urinalysis"]
        med_names = ["Warfarin", "Heparin", "Aspirin", "Plavix", "Metoprolol"]
        proc_names = ["Surgery", "Catheterization", "Biopsy", "Endoscopy", "Dialysis"]
        other_names = ["Imaging", "Cardiology_Consult", "Monitoring", "Assessment", "Care_Planning"]

        for lst in [lab_names, med_names, proc_names, other_names]:
            self.rng.shuffle(lst)

        for i, n in enumerate(self.lab_nodes):
            self.node_names[n] = lab_names[i % len(lab_names)]
            self.node_types[n] = "LAB"

        for i, n in enumerate(self.med_nodes):
            self.node_names[n] = med_names[i % len(med_names)]
            self.node_types[n] = "MEDICATION"

        for i, n in enumerate(self.proc_nodes):
            self.node_names[n] = proc_names[i % len(proc_names)]
            self.node_types[n] = "PROCEDURE"

        for i, n in enumerate(self.other_nodes):
            self.node_names[n] = other_names[i % len(other_names)]
            self.node_types[n] = "OTHER"

    def build_graph(self):
        self.build_random_graph(min_path_length=3)

        for lab in self.lab_nodes:
            if self.rng.random() < 0.7:
                if self.proc_nodes:
                    proc = self.rng.choice(self.proc_nodes)
                    self.graph.add_edge(lab, proc)
            if self.rng.random() < 0.5:
                if self.med_nodes:
                    med = self.rng.choice(self.med_nodes)
                    self.graph.add_edge(lab, med)

        for med in self.med_nodes:
            if self.rng.random() < 0.6:
                if self.proc_nodes:
                    proc = self.rng.choice(self.proc_nodes)
                    self.graph.add_edge(med, proc)

        for proc in self.proc_nodes:
            if not self.graph.has_edge(proc, self.goal_node):
                if self.rng.random() < 0.8:
                    self.graph.add_edge(proc, self.goal_node)

    # Might need clarification of selection
    def generate_constraints(self):
        cfg = {
            Difficulty.TRIVIAL: {'risk': False, 'cardinality': False, 'mutex': False, 'conditional': False},
            Difficulty.EASY: {'risk': True, 'cardinality': False, 'mutex': False, 'conditional': False},
            Difficulty.MEDIUM: {'risk': True, 'cardinality': True, 'mutex': True, 'conditional': True},
            Difficulty.HARD: {'risk': True, 'cardinality': True, 'mutex': True, 'conditional': True},
            Difficulty.ADVERSARIAL: {'risk': True, 'cardinality': True, 'mutex': True, 'conditional': True},
        }[self.difficulty]

        if cfg['risk'] and len(self.proc_nodes) >= 2:
            proc_risks = {p: self.rng.uniform(20, 40) for p in self.proc_nodes}
            interactions = [(self.proc_nodes[0], self.proc_nodes[1], 30.0)]
            # thresholds tuned so ~30% of random paths violate at medium difficulty
            threshold = {
                Difficulty.TRIVIAL: 200.0, Difficulty.EASY: 140.0,
                Difficulty.MEDIUM: 110.0, Difficulty.HARD: 90.0, Difficulty.ADVERSARIAL: 70.0
            }[self.difficulty]

            p1, p2 = self.node_names[self.proc_nodes[0]], self.node_names[self.proc_nodes[1]]
            self.constraints.append(RiskAccumulationConstraint(
                constraint_id="procedure_risk",
                description=f"Total risk < {threshold:.0f}. '{p1}' + '{p2}' add +30 extra.",
                node_risks=proc_risks, risk_threshold=threshold, interaction_pairs=interactions
            ))

        if cfg['cardinality'] and self.proc_nodes and len(self.lab_nodes) >= 3:
            surgery = self.proc_nodes[0]
            req_labs = self.lab_nodes[:3]
            min_labs = {
                Difficulty.TRIVIAL: 1, Difficulty.EASY: 1, Difficulty.MEDIUM: 2,
                Difficulty.HARD: 2, Difficulty.ADVERSARIAL: 3
            }[self.difficulty]

            lab_names = [self.node_names[l] for l in req_labs]
            self.constraints.append(CardinalityConstraint(
                constraint_id="lab_requirement",
                description=f"Before visiting '{self.node_names[surgery]}', complete >= {min_labs} of {lab_names}",
                node_set=req_labs, min_count=min_labs, max_count=len(req_labs), before_node=surgery
            ))

        if cfg['mutex']:
            thinners = [n for n in self.med_nodes
                        if any(x in self.node_names.get(n, "").lower()
                               for x in ["warfarin", "heparin", "aspirin", "plavix"])]
            if len(thinners) >= 2:
                names = [self.node_names[n] for n in thinners]
                self.constraints.append(MutexGroupConstraint(
                    constraint_id="blood_thinner_mutex",
                    description=f"Only ONE blood thinner: {names}",
                    mutex_nodes=thinners
                ))

            if self.med_nodes and self.proc_nodes:
                med = self.med_nodes[0]
                proc = self.proc_nodes[0]
                self.constraints.append(ConflictConstraint(
                    constraint_id="med_proc_conflict",
                    description=f"'{self.node_names[proc]}' contraindicated after '{self.node_names[med]}'",
                    target_node=proc, conflicting_node=med
                ))

        if self.proc_nodes:
            self.constraints.append(PrerequisiteConstraint(
                constraint_id="discharge_prereq",
                description=f"Before visiting 'Discharge', you must first visit '{self.node_names[self.proc_nodes[0]]}'",
                target_node=self.goal_node, required_nodes=[self.proc_nodes[0]]
            ))

        if cfg['conditional'] and len(self.med_nodes) >= 1 and len(self.proc_nodes) >= 2:
            condition_meds = self.med_nodes[:2]
            proc_costs = {p: self.rng.uniform(30, 50) for p in self.proc_nodes}
            limit = {
                Difficulty.MEDIUM: 150.0, Difficulty.HARD: 120.0, Difficulty.ADVERSARIAL: 90.0
            }.get(self.difficulty, 150.0)

            med_names = [self.node_names[m] for m in condition_meds]
            self.constraints.append(ConditionalCostConstraint(
                constraint_id="med_proc_cost",
                description=f"If medication ({med_names}) taken, procedure costs 1.5x. Total < {limit:.0f}",
                condition_nodes=condition_meds,
                affected_nodes=self.proc_nodes,
                base_costs=proc_costs,
                multiplier=1.5,
                total_limit=limit
            ))

    def get_action_impact(self, action: int, state: dict) -> str:
        impacts = []
        visited = set(state['history'])

        for c in self.constraints:
            if c.constraint_id == "procedure_risk" and action in c.node_risks:
                risk_add = c.node_risks[action]
                for p1, p2, extra in c.interaction_pairs:
                    if (action == p1 and p2 in visited) or (action == p2 and p1 in visited):
                        risk_add += extra
                impacts.append(f"risk +{risk_add:.0f}")

            if c.constraint_id == "med_proc_cost" and action in c.base_costs:
                cond_met = any(n in visited for n in c.condition_nodes)
                mult = c.multiplier if cond_met else 1.0
                cost = c.base_costs[action] * mult
                impacts.append(f"proc cost +{cost:.0f}")

        return ", ".join(impacts) if impacts else ""

    def get_constraint_status(self, state: dict) -> str:
        lines = []
        visited = set(state['history'])

        for c in self.constraints:
            if c.constraint_id == "procedure_risk":
                risk = sum(c.node_risks.get(n, 0) for n in visited if n in c.node_risks)
                for p1, p2, extra in c.interaction_pairs:
                    if p1 in visited and p2 in visited:
                        risk += extra
                lines.append(f"Risk: {risk:.0f}/{c.risk_threshold:.0f}")

        for c in self.constraints:
            if c.constraint_id == "med_proc_cost":
                cond_met = any(n in visited for n in c.condition_nodes)
                mult = c.multiplier if cond_met else 1.0
                cost = sum(c.base_costs.get(n, 0) * mult for n in visited if n in c.base_costs)
                lines.append(f"Procedure cost: {cost:.0f}/{c.total_limit:.0f} (multiplier: {mult}x)")

        for c in self.constraints:
            if c.constraint_id == "discharge_prereq":
                met = all(r in visited for r in c.required_nodes)
                req_names = [self.node_names.get(r, str(r)) for r in c.required_nodes]
                lines.append(f"Discharge prereq ({req_names}): {'MET' if met else 'NOT MET'}")

        for c in self.constraints:
            if c.constraint_id == "lab_requirement":
                done = sum(1 for n in c.node_set if n in visited)
                lines.append(f"Labs completed: {done}/{c.min_count} required")

        return "\n".join(lines)


class BankingEnvironment(BaseEnvironment):
    def __init__(self, num_nodes: int = 15, seed: int = 36, difficulty: Difficulty = Difficulty.MEDIUM):
        super().__init__(num_nodes, seed, difficulty)
        # banking constraints are tighter, sometimes needs more attempts
        for i in range(config.SOLVABILITY_MAX_RETRIES):
            if self.try_build(i * 1000):
                break
        else:
            raise RuntimeError(f"BankingEnvironment: no solvable config after {config.SOLVABILITY_MAX_RETRIES} tries")

    def assign_node_types(self):
        self.node_names[0] = "Login"
        self.node_types[0] = "START"
        self.node_names[self.goal_node] = "Logout_Complete"
        self.node_types[self.goal_node] = "GOAL"

        remaining = list(range(1, self.num_nodes - 1))
        self.rng.shuffle(remaining)

        n_verify = max(2, len(remaining) // 4)
        n_transfer = max(3, len(remaining) // 3)

        self._verify_nodes = remaining[:n_verify]
        self._transfer_nodes = remaining[n_verify:n_verify + n_transfer]
        self._other_nodes = remaining[n_verify + n_transfer:]

        verify_names = ["2FA_Verification", "Security_Questions", "Manager_Approval", "Biometric_Scan"]
        amounts = [5000, 8000, 10000, 3000, 12000, 7000]
        other_names = ["Check_Balance", "View_History", "Account_Settings", "Customer_Support"]

        self.rng.shuffle(verify_names)
        self.rng.shuffle(amounts)

        self._transfer_costs = {}
        for i, n in enumerate(self._verify_nodes):
            self.node_names[n] = verify_names[i % len(verify_names)]
            self.node_types[n] = "VERIFICATION"

        for i, n in enumerate(self._transfer_nodes):
            amt = amounts[i % len(amounts)]
            self.node_names[n] = f"Transfer_${amt:,}"
            self.node_types[n] = "TRANSFER"
            self._transfer_costs[n] = float(amt)

        for i, n in enumerate(self._other_nodes):
            self.node_names[n] = other_names[i % len(other_names)]
            self.node_types[n] = "OTHER"

    def build_graph(self):
        base_min = {
            Difficulty.TRIVIAL: 3, Difficulty.EASY: 3, Difficulty.MEDIUM: 4,
            Difficulty.HARD: 5, Difficulty.ADVERSARIAL: 6
        }[self.difficulty]
        extra = self.rng.randint(0, 2)
        self.build_random_graph(min_path_length=base_min + extra)

        for v in self._verify_nodes:
            self.graph.add_edge(self.start_node, v)

        for v in self._verify_nodes:
            n_transfers = self.rng.randint(1, max(1, len(self._transfer_nodes)))
            targets = self.rng.sample(self._transfer_nodes, min(n_transfers, len(self._transfer_nodes)))
            for t in targets:
                self.graph.add_edge(v, t)

        for t in self._transfer_nodes:
            n_chains = self.rng.randint(0, 2)
            other_transfers = [x for x in self._transfer_nodes if x != t]
            if other_transfers and n_chains > 0:
                targets = self.rng.sample(other_transfers, min(n_chains, len(other_transfers)))
                for t2 in targets:
                    self.graph.add_edge(t, t2)

        for t in self._transfer_nodes:
            if self.rng.random() < 0.4:
                self.graph.add_edge(t, self.goal_node)

    def generate_constraints(self):
        total_transfer_value = sum(self._transfer_costs.values())
        avg_transfer = total_transfer_value / max(1, len(self._transfer_costs))
        sorted_costs = sorted(self._transfer_costs.values())

        limit_multiplier = {
            Difficulty.TRIVIAL: 4.5, Difficulty.EASY: 3.5, Difficulty.MEDIUM: 3.0,
            Difficulty.HARD: 2.5, Difficulty.ADVERSARIAL: 2.0
        }[self.difficulty]
        limit = avg_transfer * limit_multiplier

        self.constraints.append(NumericalLimitConstraint(
            constraint_id="daily_transfer_limit",
            description=f"Daily limit ${limit:,.0f} (cumulative)",
            resource_name="transfers", limit=float(limit), node_costs=self._transfer_costs
        ))

        balance_multiplier = {
            Difficulty.TRIVIAL: 4.0, Difficulty.EASY: 3.5, Difficulty.MEDIUM: 3.0,
            Difficulty.HARD: 2.5, Difficulty.ADVERSARIAL: 2.0
        }[self.difficulty]
        balance = avg_transfer * balance_multiplier

        effects = {n: {"balance": -cost} for n, cost in self._transfer_costs.items()}
        self.constraints.append(ResourceBalanceConstraint(
            constraint_id="account_balance",
            description=f"Balance >= $0 (starting: ${balance:,.0f})",
            initial_balances={"balance": balance},
            minimum_balances={"balance": 0.0},
            node_effects=effects
        ))

        if self._verify_nodes:
            self.constraints.append(PrerequisiteConstraint(
                constraint_id="logout_verification",
                description=f"Before 'Logout', you must complete '{self.node_names[self._verify_nodes[0]]}'",
                target_node=self.goal_node, required_nodes=[self._verify_nodes[0]]
            ))

        min_transfers = {
            Difficulty.TRIVIAL: 1, Difficulty.EASY: 1, Difficulty.MEDIUM: 2,
            Difficulty.HARD: 2, Difficulty.ADVERSARIAL: 3
        }[self.difficulty]
        min_transfers = min(min_transfers, len(self._transfer_nodes))

        if self._transfer_nodes:
            self.constraints.append(CardinalityConstraint(
                constraint_id="minimum_activity",
                description=f"Before 'Logout', complete >= {min_transfers} transfer(s)",
                node_set=self._transfer_nodes, min_count=min_transfers,
                max_count=len(self._transfer_nodes),
                before_node=self.goal_node
            ))

        high_value_threshold = sorted_costs[len(sorted_costs) // 2] if sorted_costs else 8000
        high_value = [n for n, cost in self._transfer_costs.items() if cost >= high_value_threshold]
        for i, hv in enumerate(high_value[:2]):
            if self._verify_nodes:
                self.constraints.append(PrerequisiteConstraint(
                    constraint_id=f"high_value_verification_{i}",
                    description=f"Before '{self.node_names[hv]}', complete verification first",
                    target_node=hv, required_nodes=[self._verify_nodes[0]]
                ))

        if self.difficulty in [Difficulty.MEDIUM, Difficulty.HARD, Difficulty.ADVERSARIAL]:
            if high_value and len(self._transfer_nodes) >= 3:
                other_transfers = [n for n in self._transfer_nodes if n not in high_value]
                if other_transfers:
                    compliance_costs = {n: self._transfer_costs[n] * 0.1 for n in other_transfers}
                    compliance_limit = avg_transfer * 0.3 * {
                        Difficulty.MEDIUM: 1.0, Difficulty.HARD: 0.8, Difficulty.ADVERSARIAL: 0.6
                    }.get(self.difficulty, 1.0)

                    hv_names = [self.node_names[h] for h in high_value[:1]]
                    self.constraints.append(ConditionalCostConstraint(
                        constraint_id="compliance_overhead",
                        description=f"If high-value transfer ({hv_names}) done, other transfers cost 1.5x compliance. Total < ${compliance_limit:,.0f}",
                        condition_nodes=high_value[:1],
                        affected_nodes=other_transfers,
                        base_costs=compliance_costs,
                        multiplier=1.5,
                        total_limit=compliance_limit
                    ))

    def get_action_impact(self, action: int, state: dict) -> str:
        impacts = []

        if action in self._transfer_costs:
            amt = self._transfer_costs[action]
            impacts.append(f"transfers +${amt:,.0f}")
            impacts.append(f"balance -${amt:,.0f}")

        return ", ".join(impacts) if impacts else ""

    def get_constraint_status(self, state: dict) -> str:
        lines = []
        visited = set(state['history'])

        for c in self.constraints:
            if c.constraint_id == "daily_transfer_limit":
                used = sum(c.node_costs.get(n, 0) for n in visited if n in c.node_costs)
                lines.append(f"Transfers: ${used:,.0f}/${c.limit:,.0f}")

        for c in self.constraints:
            if c.constraint_id == "account_balance":
                spent = sum(abs(c.node_effects.get(n, {}).get('balance', 0)) for n in visited if n in c.node_effects)
                remaining = c.initial_balances['balance'] - spent
                lines.append(f"Balance: ${remaining:,.0f} remaining")

        for c in self.constraints:
            if c.constraint_id == "minimum_activity":
                done = sum(1 for n in c.node_set if n in visited)
                lines.append(f"Transfers completed: {done}/{c.min_count} required")

        for c in self.constraints:
            if c.constraint_id == "logout_verification":
                met = all(r in visited for r in c.required_nodes)
                lines.append(f"Verification: {'DONE' if met else 'PENDING'}")

        return "\n".join(lines)


class FileSystemEnvironment(BaseEnvironment):
    def __init__(self, num_nodes: int = 15, seed: int = 36, difficulty: Difficulty = Difficulty.MEDIUM):
        super().__init__(num_nodes, seed, difficulty)
        for attempt in range(config.SOLVABILITY_MAX_RETRIES):
            if self.try_build(attempt * 1000):
                break
        else:
            raise RuntimeError(f"FileSystemEnvironment unsolvable (seed={seed})")

    def assign_node_types(self):
        self.node_names[0] = "Session_Start"
        self.node_types[0] = "START"
        self.node_names[self.goal_node] = "Session_Complete"
        self.node_types[self.goal_node] = "GOAL"

        remaining = list(range(1, self.num_nodes - 1))
        self.rng.shuffle(remaining)

        n_read = max(2, len(remaining) // 4)
        n_write = max(2, len(remaining) // 4)
        n_backup = max(2, len(remaining) // 4)
        n_delete = max(1, len(remaining) // 5)

        self._read_nodes = remaining[:n_read]
        self._write_nodes = remaining[n_read:n_read + n_write]
        self._backup_nodes = remaining[n_read + n_write:n_read + n_write + n_backup]
        self._delete_nodes = remaining[n_read + n_write + n_backup:n_read + n_write + n_backup + n_delete]
        self._other_nodes = remaining[n_read + n_write + n_backup + n_delete:]

        files = ["config.yaml", "database.db", "user_data.json", "logs.txt", "cache.tmp"]
        self.rng.shuffle(files)

        self._op_costs = {}
        for i, n in enumerate(self._read_nodes):
            self.node_names[n] = f"Read_{files[i % len(files)]}"
            self.node_types[n] = "READ"
            self._op_costs[n] = 10.0

        for i, n in enumerate(self._write_nodes):
            self.node_names[n] = f"Write_{files[i % len(files)]}"
            self.node_types[n] = "WRITE"
            self._op_costs[n] = 30.0

        for i, n in enumerate(self._backup_nodes):
            self.node_names[n] = f"Backup_{files[i % len(files)]}"
            self.node_types[n] = "BACKUP"
            self._op_costs[n] = 50.0

        for i, n in enumerate(self._delete_nodes):
            self.node_names[n] = f"Delete_{files[i % len(files)]}"
            self.node_types[n] = "DELETE"
            self._op_costs[n] = 5.0

        for i, n in enumerate(self._other_nodes):
            self.node_names[n] = f"Check_Status_{i}"
            self.node_types[n] = "STATUS"
            self._op_costs[n] = 2.0

    def build_graph(self):
        self.build_random_graph(min_path_length=3)

        for r in self._read_nodes:
            n_writes = self.rng.randint(1, max(1, len(self._write_nodes) // 2))
            targets = self.rng.sample(self._write_nodes, min(n_writes, len(self._write_nodes)))
            for w in targets:
                self.graph.add_edge(r, w)

        for w in self._write_nodes:
            if self.rng.random() < 0.6:
                if self._backup_nodes:
                    b = self.rng.choice(self._backup_nodes)
                    self.graph.add_edge(w, b)

        for b in self._backup_nodes:
            if self.rng.random() < 0.5 and self._delete_nodes:
                d = self.rng.choice(self._delete_nodes)
                self.graph.add_edge(b, d)
            if self.rng.random() < 0.6:
                self.graph.add_edge(b, self.goal_node)

    def generate_constraints(self):
        total_io_cost = sum(self._op_costs.values())
        budget_ratio = {
            Difficulty.TRIVIAL: 0.9, Difficulty.EASY: 0.7, Difficulty.MEDIUM: 0.6,
            Difficulty.HARD: 0.5, Difficulty.ADVERSARIAL: 0.4
        }[self.difficulty]
        budget = total_io_cost * budget_ratio

        self.constraints.append(NumericalLimitConstraint(
            constraint_id="io_budget",
            description=f"I/O cost < {budget:.0f} (Read=10, Write=30, Backup=50, Delete=5)",
            resource_name="io", limit=budget, node_costs=self._op_costs
        ))

        if self._backup_nodes:
            self.constraints.append(PrerequisiteConstraint(
                constraint_id="complete_backup",
                description=f"Before 'Session_Complete', you must visit '{self.node_names[self._backup_nodes[0]]}'",
                target_node=self.goal_node, required_nodes=[self._backup_nodes[0]]
            ))

        if self.difficulty in [Difficulty.MEDIUM, Difficulty.HARD, Difficulty.ADVERSARIAL]:
            if self._write_nodes and self._read_nodes:
                min_reads = {
                    Difficulty.MEDIUM: 1, Difficulty.HARD: 1, Difficulty.ADVERSARIAL: 2
                }[self.difficulty]
                min_reads = min(min_reads, len(self._read_nodes))

                read_names = [self.node_names[r] for r in self._read_nodes[:3]]
                self.constraints.append(CardinalityConstraint(
                    constraint_id="read_before_write",
                    description=f"Before writing, complete >= {min_reads} read(s) from: {read_names}",
                    node_set=self._read_nodes[:3], min_count=min_reads,
                    before_node=self._write_nodes[0]
                ))

        if self._delete_nodes and len(self._backup_nodes) >= 2:
            min_backups = {
                Difficulty.TRIVIAL: 1, Difficulty.EASY: 1, Difficulty.MEDIUM: 1,
                Difficulty.HARD: 2, Difficulty.ADVERSARIAL: 2
            }[self.difficulty]
            min_backups = min(min_backups, len(self._backup_nodes))

            backup_names = [self.node_names[b] for b in self._backup_nodes[:3]]
            self.constraints.append(CardinalityConstraint(
                constraint_id="backup_before_delete",
                description=f"Before deleting, complete >= {min_backups} backup(s) from: {backup_names}",
                node_set=self._backup_nodes[:3], min_count=min_backups,
                before_node=self._delete_nodes[0]
            ))

        if self.difficulty in [Difficulty.HARD, Difficulty.ADVERSARIAL]:
            if self._delete_nodes and self._write_nodes:
                write_costs = {w: self._op_costs.get(w, 30.0) for w in self._write_nodes}
                io_limit = sum(write_costs.values()) * {
                    Difficulty.HARD: 0.8, Difficulty.ADVERSARIAL: 0.6
                }.get(self.difficulty, 0.8)

                del_names = [self.node_names[d] for d in self._delete_nodes[:1]]
                self.constraints.append(ConditionalCostConstraint(
                    constraint_id="cautious_write",
                    description=f"If delete ({del_names}) done, writes cost 1.5x I/O. Total < {io_limit:.0f}",
                    condition_nodes=self._delete_nodes[:1],
                    affected_nodes=self._write_nodes,
                    base_costs=write_costs,
                    multiplier=1.5,
                    total_limit=io_limit
                ))

    def get_action_impact(self, action: int, state: dict) -> str:
        if action in self._op_costs:
            cost = self._op_costs[action]
            return f"I/O +{cost:.0f}"
        return ""

    def get_constraint_status(self, state: dict) -> str:
        lines = []
        visited = set(state['history'])

        for c in self.constraints:
            if c.constraint_id == "io_budget":
                used = sum(c.node_costs.get(n, 0) for n in visited if n in c.node_costs)
                lines.append(f"I/O used: {used:.0f}/{c.limit:.0f}")

        for c in self.constraints:
            if c.constraint_id == "complete_backup":
                met = all(r in visited for r in c.required_nodes)
                req_names = [self.node_names.get(r, str(r)) for r in c.required_nodes]
                lines.append(f"Required backup ({req_names}): {'DONE' if met else 'PENDING'}")

        for c in self.constraints:
            if c.constraint_id == "read_before_write":
                done = sum(1 for n in c.node_set if n in visited)
                lines.append(f"Reads completed: {done}/{c.min_count} required before write")

        for c in self.constraints:
            if c.constraint_id == "backup_before_delete":
                done = sum(1 for n in c.node_set if n in visited)
                lines.append(f"Backups completed: {done}/{c.min_count} required before delete")

        return "\n".join(lines)


def create_environment(domain: str, num_nodes: int = None, seed: int = 36,
                       difficulty: str = "medium") -> BaseEnvironment:
    diff = Difficulty(difficulty.lower())
    if num_nodes is None:
        num_nodes = config.get_num_nodes(difficulty.lower())
    if domain == "medical":
        return MedicalEnvironment(num_nodes, seed, diff)
    elif domain == "banking":
        return BankingEnvironment(num_nodes, seed, diff)
    elif domain == "filesystem":
        return FileSystemEnvironment(num_nodes, seed, diff)
    raise ValueError(f"Unknown domain: {domain}")


def is_solvable(env: BaseEnvironment, max_depth: int = None,
                      max_iter: int = None, timeout: float = None) -> Tuple[bool, int]:
    if max_depth is None:
        max_depth = config.SOLVABILITY_MAX_DEPTH
    if max_iter is None:
        max_iter = config.SOLVABILITY_MAX_ITER
    if timeout is None:
        timeout = config.SOLVABILITY_TIMEOUT
    start = time.time()
    initial = env.reset()
    queue = deque([(initial, 0)])
    visited = {(initial['current_loc'], tuple(initial['history']))}

    iters = 0
    while queue:
        iters += 1
        if time.time() - start >= timeout or iters >= max_iter:
            return False, -1

        state, depth = queue.popleft()
        if depth > max_depth:
            continue
        if env.is_goal_reached(state):
            return True, depth

        for action in env.get_available_actions(state):
            new_state = env.transition(state, action)
            is_safe, _ = env.check_safety(new_state)
            if is_safe:
                key = (new_state['current_loc'], tuple(new_state['history']))
                if key not in visited:
                    visited.add(key)
                    queue.append((new_state, depth + 1))

    return False, -1


if __name__ == "__main__":
    for domain in ["medical", "banking", "filesystem"]:
        print(f"\n[{domain.upper()}]")
        for diff in ["easy", "medium", "hard"]:
            env = create_environment(domain, num_nodes=15, seed=36, difficulty=diff)
            solvable, steps = is_solvable(env)
            print(f"  {diff}: {len(env.constraints)} constraints, solvable={solvable}, steps={steps}")
