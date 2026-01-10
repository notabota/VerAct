import argparse
import json
import logging
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from src import veract_agent

import pandas as pd
from tqdm import tqdm

from src import config
from src.environments import create_environment, check_solvability
from src.baselines import (
    run_oracle, run_random_safe, run_react, run_llm_check, run_veract,
    run_react_cot, run_react_con, run_code_check
)
from src.benchmark_logger import BenchmarkLogger, EpisodeLogger
from src.stats_utils import print_statistical_summary


@dataclass
class BenchmarkResult:
    method: str
    domain: str
    difficulty: str
    seed: int
    status: str
    steps: int
    optimal_steps: int
    llm_calls: int
    tokens_input: int
    tokens_output: int
    verifier_calls: int
    violations: int
    false_negatives: int
    false_positives: int
    proposals_generated: int
    proposals_valid: int
    proposals_safe: int
    num_nodes: int
    num_constraints: int
    wall_time: float
    path: List[int]


ALL_METHODS = {
    "Oracle": run_oracle,
    "Random-Safe": run_random_safe,
    "ReAct": run_react,
    "ReAct-CoT": run_react_cot,
    "ReAct-Conservative": run_react_con,
    "LLM-Check": run_llm_check,
    "Code-Check": run_code_check,
    "VerAct": run_veract,
}


def run_single_task(task_info: dict) -> Tuple[Optional[BenchmarkResult], Optional[dict]]:
    domain = task_info["domain"]
    difficulty = task_info["difficulty"]
    seed = task_info["seed"]
    method = task_info["method"]
    optimal = task_info["optimal"]
    save_traces = task_info["save_traces"]
    n_candidates = task_info.get("n_candidates", config.N_CANDIDATES)
    max_retries = task_info.get("max_retries", config.MAX_RETRIES)

    try:
        env = create_environment(domain, config.get_num_nodes(difficulty), seed, difficulty)
        episode_logger = EpisodeLogger() if save_traces else None

        method_start = time.time()

        if method == "Random-Safe":
            baseline_result = ALL_METHODS[method](env, seed=seed, logger=episode_logger)
        elif method == "VerAct":
            baseline_result = ALL_METHODS[method](env, n_candidates=n_candidates, max_retries=max_retries, logger=episode_logger)
        elif method in ("LLM-Check", "Code-Check"):
            baseline_result = ALL_METHODS[method](env, max_retries=max_retries, logger=episode_logger)
        else:
            baseline_result = ALL_METHODS[method](env, logger=episode_logger)

        method_time = time.time() - method_start

        result = BenchmarkResult(
            method=method,
            domain=domain,
            difficulty=difficulty,
            seed=seed,
            status=baseline_result.status,
            steps=baseline_result.steps,
            optimal_steps=optimal,
            llm_calls=baseline_result.llm_calls,
            tokens_input=baseline_result.tokens_input,
            tokens_output=baseline_result.tokens_output,
            verifier_calls=baseline_result.verifier_calls,
            violations=baseline_result.violations,
            false_negatives=baseline_result.false_negatives,
            false_positives=baseline_result.false_positives,
            proposals_generated=baseline_result.proposals_generated,
            proposals_valid=baseline_result.proposals_valid,
            proposals_safe=baseline_result.proposals_safe,
            num_nodes=env.num_nodes,
            num_constraints=len(env.constraints),
            wall_time=method_time,
            path=baseline_result.path
        )

        trace = None
        if episode_logger:
            trace = episode_logger.finish(
                status=baseline_result.status,
                path=baseline_result.path,
                stats={
                    "optimal_steps": optimal,
                    "llm_calls": baseline_result.llm_calls,
                    "tokens_input": baseline_result.tokens_input,
                    "tokens_output": baseline_result.tokens_output,
                    "verifier_calls": baseline_result.verifier_calls,
                    "violations": baseline_result.violations,
                    "false_positives": baseline_result.false_positives,
                    "false_negatives": baseline_result.false_negatives,
                    "wall_time": method_time
                }
            )

        return result, trace

    except Exception as e:
        logging.getLogger(__name__).error(f"Error {domain}/{difficulty}/s{seed}/{method}: {e}")
        return None, None


def run_benchmark(
    domains: List[str] = None,
    difficulties: List[str] = None,
    methods: List[str] = None,
    n_seeds: int = None,
    output_dir: str = None,
    verbose: bool = True,
    log_level: str = "INFO",
    save_traces: bool = True,
    workers: int = 1
) -> Tuple[List[BenchmarkResult], pd.DataFrame]:
    domains = domains or config.DOMAINS
    difficulties = difficulties or config.DIFFICULTIES
    methods = methods or list(ALL_METHODS.keys())
    n_seeds = n_seeds or config.NUM_SEEDS
    output_dir = output_dir or config.OUTPUT_DIR

    benchmark_config = {
        "domains": domains,
        "difficulties": difficulties,
        "methods": methods,
        "num_seeds": n_seeds,
        "max_depth": config.MAX_DEPTH,
        "n_candidates": config.N_CANDIDATES,
        "max_retries": config.MAX_RETRIES,
        "temperature": config.TEMPERATURE,
        "provider": config.LLM_PROVIDER,
        "model": config.MODEL,
        "model_id": config.get_model_id()
    }

    benchmark_logger = BenchmarkLogger(output_dir, benchmark_config) if save_traces else None
    run_output_dir = benchmark_logger.get_output_dir() if benchmark_logger else Path(output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_output_dir / "run.log"
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file)],
        force=True
    )
    logger = logging.getLogger(__name__)
    logger.info("VerAct benchmark starting")

    invalid = set(methods) - set(ALL_METHODS.keys())
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}. Available: {list(ALL_METHODS.keys())}")

    total_runs = len(domains) * len(difficulties) * len(methods) * n_seeds
    logger.info(f"Configuration: {domains} x {difficulties} x {methods} x {n_seeds} seeds = {total_runs} runs")

    if verbose:
        print(f"VerAct Benchmark")
        print(f"Output: {run_output_dir}")
        print(f"Config: {domains} x {difficulties} x {methods} x {n_seeds} seeds = {total_runs} runs")
        print(f"Workers: {workers} | k={config.N_CANDIDATES} | retries={config.MAX_RETRIES}", flush=True)

    results = []
    start_time = time.time()

    env_info = {}
    env_checks = [(d, diff, s) for d in domains for diff in difficulties for s in range(n_seeds)]
    total_envs = len(env_checks)

    for i, (domain, difficulty, seed) in enumerate(env_checks):
        if verbose:
            print(f"\rChecking environments: {i+1}/{total_envs}" + " " * 10, end="", flush=True)
        env_key = (domain, difficulty, seed)
        try:
            env = create_environment(domain, config.get_num_nodes(difficulty), seed, difficulty)
            solvable, optimal = check_solvability(env, max_depth=config.MAX_DEPTH)
            if solvable:
                env_info[env_key] = optimal
            else:
                logger.warning(f"Unsolvable: {domain}/{difficulty}/s{seed}")
        except Exception as e:
            logger.error(f"Environment error {domain}/{difficulty}/s{seed}: {e}")

    if verbose:
        print(f"\rChecking environments: {total_envs}/{total_envs} - {len(env_info)} ready" + " " * 10, flush=True)

    tasks = []
    for domain in domains:
        for difficulty in difficulties:
            for seed in range(n_seeds):
                env_key = (domain, difficulty, seed)
                if env_key not in env_info:
                    continue
                for method in methods:
                    tasks.append({
                        "domain": domain,
                        "difficulty": difficulty,
                        "seed": seed,
                        "method": method,
                        "optimal": env_info[env_key],
                        "save_traces": save_traces,
                        "n_candidates": config.N_CANDIDATES,
                        "max_retries": config.MAX_RETRIES
                    })

    if workers > 1:
        logger.info(f"Running with {workers} parallel workers")

        completed = 0
        total = len(tasks)
        status_counts = {"SUCCESS": 0, "VIOLATION": 0, "STUCK": 0, "MAX_DEPTH": 0}

        if verbose:
            print(f"Progress: 0/{total} (0%)", end="", flush=True)

        # Run 1 by 1 is insanely wrong, need parallel
        # however it cause thread related issue, need careful test
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_single_task, task): task for task in tasks}

            for future in as_completed(futures):
                try:
                    result, trace = future.result()
                    if result:
                        results.append(result)
                        if trace and benchmark_logger:
                            benchmark_logger.save_episode(trace)
                        status_counts[result.status] = status_counts.get(result.status, 0) + 1
                except Exception as e:
                    logger.error(f"Task error: {e}")

                completed += 1

                if verbose:
                    pct = completed * 100 // total
                    stats = " ".join(f"{k[0]}:{v}" for k, v in status_counts.items() if v > 0)
                    print(f"\rProgress: {completed}/{total} ({pct}%) [{stats}]" + " " * 20, end="", flush=True)

        if verbose:
            print()
    else:
        pbar = tqdm(tasks, desc="Running", unit="run", ncols=100, disable=not verbose)

        for task in pbar:
            pbar.set_description(f"{task['domain'][:3]}/{task['difficulty'][:3]}/s{task['seed']}/{task['method'][:8]}")

            result, trace = run_single_task(task)
            if result:
                results.append(result)
                if trace and benchmark_logger:
                    benchmark_logger.save_episode(trace)
                icon = "[OK]" if result.status == "SUCCESS" else "[X]" if result.status == "VIOLATION" else "[.]"
                pbar.set_postfix_str(f"{icon} {result.status[:7]}")

        pbar.close()
    total_time = time.time() - start_time

    if verbose and results:
        print(f"\nCompleted: {len(results)} runs in {total_time:.1f}s")
        status_counts = {}
        for r in results:
            status_counts[r.status] = status_counts.get(r.status, 0) + 1
        print(f"Status: {status_counts}")

    df = pd.DataFrame([asdict(r) for r in results])

    if benchmark_logger:
        benchmark_logger.save_results([asdict(r) for r in results])
        if verbose:
            print(f"Output: {benchmark_logger.get_output_dir()}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(run_output_dir / f"results_{timestamp}.csv", index=False)
        with open(run_output_dir / f"results_{timestamp}.json", 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)

    return results, df


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in df['method'].unique():
        for domain in df['domain'].unique():
            for diff in df['difficulty'].unique():
                subset = df[(df['method'] == method) & (df['domain'] == domain) & (df['difficulty'] == diff)]
                if len(subset) == 0:
                    continue
                n = len(subset)
                rows.append({
                    'method': method,
                    'domain': domain,
                    'difficulty': diff,
                    'n': n,
                    'success_rate': (subset['status'] == 'SUCCESS').sum() / n * 100,
                    'violation_rate': (subset['status'] == 'VIOLATION').sum() / n * 100,
                    'stuck_rate': (subset['status'] == 'STUCK').sum() / n * 100,
                    'mean_llm_calls': subset['llm_calls'].mean(),
                    'mean_steps': subset['steps'].mean(),
                    'mean_optimal': subset['optimal_steps'].mean(),
                })
    return pd.DataFrame(rows)


def print_results_table(df: pd.DataFrame):
    print("\nResults summary")
    if df.empty:
        print("No results to display")
        return
    summary = compute_summary(df)
    for domain in summary['domain'].unique():
        print(f"\n{domain}:")
        print(f"{'Method':<20} {'Success':>8} {'Violate':>8} {'Stuck':>8} {'LLM':>6} {'Steps':>6}")
        for _, row in summary[summary['domain'] == domain].iterrows():
            print(f"{row['method']:<20} {row['success_rate']:>7.1f}% {row['violation_rate']:>7.1f}% "
                  f"{row['stuck_rate']:>7.1f}% {row['mean_llm_calls']:>6.1f} {row['mean_steps']:>6.1f}")


def main():

    # Probably want to write a bash script to automate run
    parser = argparse.ArgumentParser(description="VerAct Benchmark Runner")
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--difficulties", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--provider", type=str, default=None,
                       choices=["azure", "bedrock", "vertex"],
                       help="LLM provider: azure, bedrock, or vertex")
    parser.add_argument("--model", type=str, default=None,
                       help="Model: gpt-4o/gpt-4o-mini (azure), claude-3.5-sonnet (bedrock), gemini-3-pro (vertex)")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-traces", action="store_true")
    parser.add_argument("--workers", "-w", type=int, default=8,
                       help="Number of parallel workers (default: 8)")
    parser.add_argument("-k", "--k-candidates", type=int, default=None,
                       help="Number of candidate proposals per step (default: 10)")
    parser.add_argument("-r", "--max-retries", type=int, default=None,
                       help="Max retries for LLM calls (default: 5)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    if args.provider:
        config.LLM_PROVIDER = args.provider
        if not args.model:
            if args.provider == "bedrock":
                config.MODEL = "claude-3.5-sonnet"
            elif args.provider == "vertex":
                config.MODEL = "gemini-3-pro"
            else:
                config.MODEL = "gpt-4o"
    if args.model:
        config.MODEL = args.model
        if not args.provider:
            if args.model in config.BEDROCK_MODELS:
                config.LLM_PROVIDER = "bedrock"
            elif args.model in config.VERTEX_MODELS:
                config.LLM_PROVIDER = "vertex"
            else:
                config.LLM_PROVIDER = "azure"

    if config.LLM_PROVIDER == "bedrock" and config.MODEL not in config.BEDROCK_MODELS:
        print(f"ERROR: Model '{config.MODEL}' not available for Bedrock. Available: {list(config.BEDROCK_MODELS.keys())}")
        return
    if config.LLM_PROVIDER == "azure" and config.MODEL not in config.AZURE_MODELS:
        print(f"ERROR: Model '{config.MODEL}' not available for Azure. Available: {list(config.AZURE_MODELS.keys())}")
        return
    if config.LLM_PROVIDER == "vertex" and config.MODEL not in config.VERTEX_MODELS:
        print(f"ERROR: Model '{config.MODEL}' not available for Vertex. Available: {list(config.VERTEX_MODELS.keys())}")
        return

    if args.provider or args.model:
        veract_agent.LLM_CLIENT = veract_agent.create_llm_client()
        print(f"Using: {config.LLM_PROVIDER} / {config.MODEL} ({config.get_model_id()})")

    if args.k_candidates is not None:
        config.N_CANDIDATES = args.k_candidates
    if args.max_retries is not None:
        config.MAX_RETRIES = args.max_retries

    if args.debug:
        args.log_level = "DEBUG"

    if args.quick:
        args.seeds = 2
        args.domains = ["medical"]
        args.difficulties = ["easy"]

    methods = args.methods
    if args.no_llm:
        methods = ["Oracle", "Random-Safe"]

    results, df = run_benchmark(
        domains=args.domains,
        difficulties=args.difficulties,
        methods=methods,
        n_seeds=args.seeds,
        output_dir=args.output,
        log_level=args.log_level,
        save_traces=not args.no_traces,
        workers=args.workers
    )

    print_results_table(df)

    if len(df) > 10:
        print_statistical_summary(df)


if __name__ == "__main__":
    main()
