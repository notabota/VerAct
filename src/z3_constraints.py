from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import time

from z3 import Solver, Bool, And, Or, Not, If, Sum, sat, unsat, BoolRef

# Consider add more if needed
# Each should represent 1 real world constraint
class ConstraintType(Enum):
    PREREQUISITE = "prerequisite"
    CONFLICT = "conflict"
    NUMERICAL_LIMIT = "numerical_limit"
    CONDITIONAL_COST = "conditional_cost"
    CARDINALITY = "cardinality"
    MUTEX_GROUP = "mutex_group"
    RISK_ACCUMULATION = "risk_accumulation"
    RESOURCE_BALANCE = "resource_balance"


@dataclass
class BaseConstraint:
    constraint_id: str
    description: str
    constraint_type: ConstraintType = field(default=ConstraintType.PREREQUISITE)

    def encode_z3(self, ctx: 'Z3Context') -> BoolRef:
        raise NotImplementedError

    def get_variables(self) -> List[str]:
        raise NotImplementedError


@dataclass
class PrerequisiteConstraint(BaseConstraint):
    target_node: int = 0
    required_nodes: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.constraint_type = ConstraintType.PREREQUISITE

    def encode_z3(self, ctx):
        if ctx.current_node != self.target_node:
            return True
        prereqs = [ctx.visited[r] for r in self.required_nodes if r in ctx.visited]
        return And(prereqs) if prereqs else True

    def get_variables(self):
        return [f"v_{self.target_node}"] + [f"v_{r}" for r in self.required_nodes]


@dataclass
class ConflictConstraint(BaseConstraint):
    target_node: int = 0
    conflicting_node: int = 0

    def __post_init__(self):
        self.constraint_type = ConstraintType.CONFLICT

    def encode_z3(self, ctx):
        if ctx.current_node != self.target_node:
            return True
        if self.conflicting_node not in ctx.visited:
            return True
        return Not(ctx.visited[self.conflicting_node])

    def get_variables(self):
        return [f"v_{self.target_node}", f"v_{self.conflicting_node}"]


@dataclass
class NumericalLimitConstraint(BaseConstraint):
    resource_name: str = ""
    limit: float = 0.0
    node_costs: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        self.constraint_type = ConstraintType.NUMERICAL_LIMIT

    def encode_z3(self, ctx):
        terms = [If(ctx.visited[n], c, 0.0) for n, c in self.node_costs.items() if n in ctx.visited]
        if not terms:
            return True
        # Z3 Sum([x]) != x in some versions, causes type errors
        total = Sum(terms) if len(terms) > 1 else terms[0]
        return total <= self.limit

    def get_variables(self):
        return [f"v_{n}" for n in self.node_costs]


@dataclass
class ConditionalCostConstraint(BaseConstraint):
    condition_nodes: List[int] = field(default_factory=list)
    affected_nodes: List[int] = field(default_factory=list)
    base_costs: Dict[int, float] = field(default_factory=dict)
    multiplier: float = 2.0
    total_limit: float = 100.0

    def __post_init__(self):
        self.constraint_type = ConstraintType.CONDITIONAL_COST

    def encode_z3(self, ctx: 'Z3Context') -> BoolRef:
        cond_terms = [ctx.visited[c] for c in self.condition_nodes if c in ctx.visited]
        cond_met = Or(cond_terms) if cond_terms else False

        cost_terms = []
        for n in self.affected_nodes:
            if n in ctx.visited and n in self.base_costs:
                base = self.base_costs[n]
                cost = If(cond_met, base * self.multiplier, base)
                cost_terms.append(If(ctx.visited[n], cost, 0.0))

        if not cost_terms:
            return True
        total = Sum(cost_terms) if len(cost_terms) > 1 else cost_terms[0]
        return total <= self.total_limit

    def get_variables(self) -> List[str]:
        nodes = set(self.condition_nodes) | set(self.affected_nodes)
        return [f"v_{n}" for n in nodes]


@dataclass
class CardinalityConstraint(BaseConstraint):
    node_set: List[int] = field(default_factory=list)
    min_count: int = 0
    max_count: int = 999
    before_node: Optional[int] = None

    def __post_init__(self):
        self.constraint_type = ConstraintType.CARDINALITY

    def encode_z3(self, ctx: 'Z3Context') -> BoolRef:
        if self.before_node is not None and ctx.current_node != self.before_node:
            return True
        count_terms = [If(ctx.visited[n], 1, 0) for n in self.node_set if n in ctx.visited]
        if not count_terms:
            return self.min_count <= 0
        count = Sum(count_terms) if len(count_terms) > 1 else count_terms[0]
        return And(count >= self.min_count, count <= self.max_count)

    def get_variables(self) -> List[str]:
        return [f"v_{n}" for n in self.node_set]


@dataclass
class MutexGroupConstraint(BaseConstraint):
    mutex_nodes: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.constraint_type = ConstraintType.MUTEX_GROUP

    def encode_z3(self, ctx: 'Z3Context') -> BoolRef:
        count_terms = [If(ctx.visited[n], 1, 0) for n in self.mutex_nodes if n in ctx.visited]
        if len(count_terms) <= 1:
            return True
        return Sum(count_terms) <= 1

    def get_variables(self) -> List[str]:
        return [f"v_{n}" for n in self.mutex_nodes]


@dataclass
class RiskAccumulationConstraint(BaseConstraint):
    node_risks: Dict[int, float] = field(default_factory=dict)
    risk_threshold: float = 100.0
    interaction_pairs: List[Tuple[int, int, float]] = field(default_factory=list)

    def __post_init__(self):
        self.constraint_type = ConstraintType.RISK_ACCUMULATION

    def encode_z3(self, ctx):
        base = [If(ctx.visited[n], r, 0.0) for n, r in self.node_risks.items() if n in ctx.visited]
        interact = []
        for n1, n2, mult in self.interaction_pairs:
            if n1 in ctx.visited and n2 in ctx.visited:
                interact.append(If(And(ctx.visited[n1], ctx.visited[n2]), mult, 0.0))
        all_terms = base + interact
        if not all_terms:
            return True
        return (Sum(all_terms) if len(all_terms) > 1 else all_terms[0]) <= self.risk_threshold

    def get_variables(self):
        nodes = set(self.node_risks.keys())
        for n1, n2, _ in self.interaction_pairs:
            nodes.add(n1)
            nodes.add(n2)
        return [f"v_{n}" for n in nodes]


@dataclass
class ResourceBalanceConstraint(BaseConstraint):
    initial_balances: Dict[str, float] = field(default_factory=dict)
    minimum_balances: Dict[str, float] = field(default_factory=dict)
    node_effects: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        self.constraint_type = ConstraintType.RESOURCE_BALANCE

    def encode_z3(self, ctx: 'Z3Context') -> BoolRef:
        constraints = []
        for resource, initial in self.initial_balances.items():
            effect_terms = []
            for node, effects in self.node_effects.items():
                if node in ctx.visited and resource in effects:
                    delta = effects[resource]
                    effect_terms.append(If(ctx.visited[node], delta, 0.0))

            if effect_terms:
                total_effect = Sum(effect_terms) if len(effect_terms) > 1 else effect_terms[0]
                final = initial + total_effect
            else:
                final = initial

            minimum = self.minimum_balances.get(resource, 0.0)
            constraints.append(final >= minimum)

        return And(constraints) if constraints else True

    def get_variables(self) -> List[str]:
        return [f"v_{n}" for n in self.node_effects.keys()]


class Z3Context:
    def __init__(self, current_node: int, history: Set[int], num_nodes: int = 100):
        self.current_node = current_node
        self.history = history
        self.num_nodes = num_nodes
        self.visited: Dict[int, BoolRef] = {i: Bool(f"v_{i}") for i in range(num_nodes)}
        self.constraint_map: Dict[str, BaseConstraint] = {}

    def ground_state(self, s: Solver):
        for i in range(self.num_nodes):
            s.add(self.visited[i] == (i in self.history))


@dataclass
class Z3VerificationResult:
    is_safe: bool
    status: str
    reason: str
    violated_constraints: List[str]
    unsat_core: List[str]
    solve_time_ms: float
    z3_assertions: int = 0
    z3_variables: int = 0


class Z3ConstraintVerifier:
    # 5s is generous, most solve in <100ms even for hard constraints
    def __init__(self, timeout_ms=5000):
        self.timeout_ms = timeout_ms

    def verify(self, state, constraints, num_nodes=100):
        t0 = time.time()
        ctx = Z3Context(state['current_loc'], set(state['history']), num_nodes)
        s = Solver()
        s.set("timeout", self.timeout_ms)
        ctx.ground_state(s)

        tracked = {}
        for c in constraints:
            try:
                formula = c.encode_z3(ctx)
                if formula is True:
                    continue
                # assert_and_track lets us get unsat_core to identify which constraint failed
                tracker = Bool(f"t_{c.constraint_id}")
                tracked[str(tracker)] = c
                s.assert_and_track(formula, tracker)
            except Exception as e:
                return Z3VerificationResult(
                    is_safe=False, status="error", reason=str(e),
                    violated_constraints=[c.constraint_id], unsat_core=[c.description],
                    solve_time_ms=(time.time()-t0)*1000
                )

        res = s.check()
        elapsed = (time.time()-t0)*1000

        if res == sat:
            return Z3VerificationResult(
                is_safe=True, status="sat", reason="OK",
                violated_constraints=[], unsat_core=[], solve_time_ms=elapsed,
                z3_assertions=len(s.assertions()), z3_variables=len(ctx.visited)
            )
        if res == unsat:
            # Core is sometimes empty
            # no idea why, just handle it
            core = s.unsat_core()
            violated, descs = [], []
            for t in core:
                if str(t) in tracked:
                    c = tracked[str(t)]
                    violated.append(c.constraint_id)
                    descs.append(c.description)
            return Z3VerificationResult(
                is_safe=False, status="unsat",
                reason=descs[0] if descs else "Violation",
                violated_constraints=violated, unsat_core=descs if descs else ["Unknown"],
                solve_time_ms=elapsed, z3_assertions=len(s.assertions()), z3_variables=len(ctx.visited)
            )
        return Z3VerificationResult(
            is_safe=False, status="unknown", reason="Timeout",
            violated_constraints=[], unsat_core=["Timeout"], solve_time_ms=elapsed
        )

    def get_safe_actions(self, state, available, constraints, num_nodes=100):
        return [(a, self.verify({'current_loc': a, 'history': state['history'] + [a]}, constraints, num_nodes))
                for a in available]

    def explain_unsafe(self, state, constraints, num_nodes=100):
        r = self.verify(state, constraints, num_nodes)
        if r.is_safe:
            return "Safe"
        return f"VIOLATION at {state['current_loc']}: {r.unsat_core}"
