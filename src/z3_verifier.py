from .z3_constraints import (
    Z3ConstraintVerifier,
    Z3VerificationResult,
)
from .environments import BaseEnvironment
from typing import List


class Z3Verifier:
    def __init__(self, timeout_ms: int = 5000):
        self.timeout_ms = timeout_ms
        self._verifier = Z3ConstraintVerifier(timeout_ms=timeout_ms)

    def verify(self, state: dict, env: BaseEnvironment) -> Z3VerificationResult:
        constraints = getattr(env, 'constraints', [])
        num_nodes = getattr(env, 'num_nodes', 100)

        return self._verifier.verify(state, constraints, num_nodes)

    def verify_action(
        self,
        state: dict,
        action: int,
        env: BaseEnvironment
    ) -> Z3VerificationResult:
        next_state = {
            'current_loc': action,
            'history': state['history'] + [action]
        }

        return self.verify(next_state, env)

    def get_safe_actions(
        self,
        state: dict,
        env: BaseEnvironment
    ) -> List[tuple]:
        results = []
        available = env.get_available_actions(state)

        for action in available:
            result = self.verify_action(state, action, env)
            results.append((action, result))

        return results

    def explain_unsafe(self, state: dict, env: BaseEnvironment) -> str:
        result = self.verify(state, env)
        if result.is_safe:
            return "Safe"
        return f"VIOLATION at {state['current_loc']}: {result.unsat_core}"
