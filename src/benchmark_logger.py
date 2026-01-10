import json
import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from src.environments import BaseEnvironment
import pandas as pd


@dataclass
class LLMCallTrace:
    step: int
    retry: int
    prompt: str
    response_raw: str
    parsed_proposals: List[Dict]
    tokens_input: int
    tokens_output: int
    latency_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class ProposalTrace:
    step: int
    target: int
    target_name: str
    is_valid: bool
    is_safe: Optional[bool] = None
    rejection_reason: Optional[str] = None
    generated_code: Optional[str] = None


@dataclass
class StepTrace:
    step: int
    state_before: Dict
    available_actions: List[int]
    available_action_names: List[str]
    llm_calls: List[LLMCallTrace] = field(default_factory=list)
    proposals_evaluated: List[ProposalTrace] = field(default_factory=list)
    action_taken: Optional[int] = None
    action_taken_name: Optional[str] = None
    state_after: Optional[Dict] = None


@dataclass
class EnvironmentSpec:
    domain: str
    difficulty: str
    seed: int
    num_nodes: int
    node_names: Dict[int, str]
    edges: List[List[int]]
    start_node: int
    goal_node: int
    constraints: List[Dict]


@dataclass
class EpisodeTrace:
    method: str
    env_spec: EnvironmentSpec
    steps: List[StepTrace] = field(default_factory=list)
    status: str = ""
    final_path: List[int] = field(default_factory=list)
    final_path_names: List[str] = field(default_factory=list)
    optimal_steps: int = 0
    total_llm_calls: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_verifier_calls: int = 0
    violations: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    wall_time_seconds: float = 0.0


@dataclass
class RunConfig:
    timestamp: str
    python_version: str
    platform: str
    domains: List[str]
    difficulties: List[str]
    methods: List[str]
    num_seeds: int
    max_depth: int
    n_candidates: int
    max_retries: int
    temperature: float
    model: str


class EpisodeLogger:
    # Each run is expensive, log everything

    def __init__(self):
        self.env_spec: Optional[EnvironmentSpec] = None
        self.method: str = ""
        self.steps: List[StepTrace] = []
        self._current_step: Optional[StepTrace] = None
        self._step_idx = 0
        self._node_names: Dict[int, str] = {}

    def start(self, method: str, env: BaseEnvironment):
        self.method = method
        self._node_names = dict(env.node_names)
        self._step_idx = 0
        self.steps = []
        self._current_step = None

        constraints = []
        for c in env.constraints:
            cd = {
                "type": type(c).__name__,
                "id": c.constraint_id,
                "description": c.description
            }
            for attr in ['target_node', 'required_nodes', 'conflicting_node', 'limit', 'resource_name']:
                if hasattr(c, attr):
                    cd[attr] = getattr(c, attr)
            constraints.append(cd)

        self.env_spec = EnvironmentSpec(
            domain=env.__class__.__name__.replace("Environment", "").lower(),
            difficulty=env.difficulty.value if hasattr(env.difficulty, 'value') else str(env.difficulty),
            seed=env.seed,
            num_nodes=env.num_nodes,
            node_names=self._node_names,
            edges=list(env.graph.edges()),
            start_node=env.start_node,
            goal_node=env.goal_node,
            constraints=constraints
        )

    def start_step(self, state: Dict, available_actions: List[int]):
        self._current_step = StepTrace(
            step=self._step_idx,
            state_before={
                "current_loc": state["current_loc"],
                "current_loc_name": self._node_names.get(state["current_loc"], ""),
                "history": list(state["history"]),
                "history_names": [self._node_names.get(h, "") for h in state["history"]]
            },
            available_actions=list(available_actions),
            available_action_names=[self._node_names.get(a, "") for a in available_actions]
        )

    def log_llm_call(self, retry: int, prompt: str, response_raw: str,
                     proposals: List, tokens_in: int, tokens_out: int,
                     latency_ms: float, success: bool = True, error: str = None):
        if not self._current_step:
            return

        parsed = []
        for p in proposals:
            if hasattr(p, 'target'):
                parsed.append({"target": p.target, "confidence": getattr(p, 'confidence', 0.5)})
            elif isinstance(p, dict):
                parsed.append(p)

        self._current_step.llm_calls.append(LLMCallTrace(
            step=self._step_idx,
            retry=retry,
            prompt=prompt,
            response_raw=response_raw,
            parsed_proposals=parsed,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            latency_ms=latency_ms,
            success=success,
            error=error
        ))

    def log_proposal(self, target: int, is_valid: bool, is_safe: bool = None,
                     rejection_reason: str = None, generated_code: str = None):
        if not self._current_step:
            return

        self._current_step.proposals_evaluated.append(ProposalTrace(
            step=self._step_idx,
            target=target,
            target_name=self._node_names.get(target, ""),
            is_valid=is_valid,
            is_safe=is_safe,
            rejection_reason=rejection_reason,
            generated_code=generated_code
        ))

    def end_step(self, action: int, state_after: Dict):
        if not self._current_step:
            return

        self._current_step.action_taken = action
        self._current_step.action_taken_name = self._node_names.get(action, "")
        self._current_step.state_after = {
            "current_loc": state_after["current_loc"],
            "current_loc_name": self._node_names.get(state_after["current_loc"], ""),
            "history": list(state_after["history"]),
            "history_names": [self._node_names.get(h, "") for h in state_after["history"]]
        }
        self.steps.append(self._current_step)
        self._step_idx += 1
        self._current_step = None

    def skip_step(self):
        if self._current_step:
            self.steps.append(self._current_step)
        self._step_idx += 1
        self._current_step = None

    def finish(self, status: str, path: List[int], stats: Dict) -> EpisodeTrace:
        return EpisodeTrace(
            method=self.method,
            env_spec=self.env_spec,
            steps=self.steps,
            status=status,
            final_path=path,
            final_path_names=[self._node_names.get(n, "") for n in path],
            optimal_steps=stats.get("optimal_steps", 0),
            total_llm_calls=stats.get("llm_calls", 0),
            total_tokens_input=stats.get("tokens_input", 0),
            total_tokens_output=stats.get("tokens_output", 0),
            total_verifier_calls=stats.get("verifier_calls", 0),
            violations=stats.get("violations", 0),
            false_positives=stats.get("false_positives", 0),
            false_negatives=stats.get("false_negatives", 0),
            wall_time_seconds=stats.get("wall_time", 0.0)
        )


class BenchmarkLogger:
    def __init__(self, output_dir: str, config: Dict):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(output_dir) / f"run_{self.timestamp}"
        self.traces_dir = self.base_dir / "traces"
        self.env_specs_dir = self.base_dir / "environment_specs"

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(exist_ok=True)
        self.env_specs_dir.mkdir(exist_ok=True)

        self.run_config = RunConfig(
            timestamp=self.timestamp,
            python_version=sys.version.split()[0],
            platform=platform.system(),
            domains=config.get("domains", []),
            difficulties=config.get("difficulties", []),
            methods=config.get("methods", []),
            num_seeds=config.get("num_seeds", 0),
            max_depth=config.get("max_depth", 0),
            n_candidates=config.get("n_candidates", 0),
            max_retries=config.get("max_retries", 0),
            temperature=config.get("temperature", 0.0),
            model=config.get("model", "")
        )

        config_path = self.base_dir / "run_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.run_config), f, indent=2)

        self.episodes: List[EpisodeTrace] = []
        self._env_specs_saved: set = set()

    # TODO: Consider save lite data for quick review instead of manual read from the logs
    def save_episode(self, trace: EpisodeTrace):
        self.episodes.append(trace)

        env_key = f"{trace.env_spec.domain}_{trace.env_spec.difficulty}_s{trace.env_spec.seed}"
        if env_key not in self._env_specs_saved:
            env_path = self.env_specs_dir / f"{env_key}.json"
            with open(env_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(trace.env_spec), f, indent=2)
            self._env_specs_saved.add(env_key)

        trace_filename = f"{env_key}_{trace.method}.json"
        trace_path = self.traces_dir / trace_filename
        with open(trace_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(trace), f, indent=2, default=str)

    def save_results(self, results: List[Dict]):
        results_path = self.base_dir / "results.csv"
        df = pd.DataFrame(results)
        df.to_csv(results_path, index=False)

        json_path = self.base_dir / "results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

    def get_output_dir(self) -> Path:
        return self.base_dir
