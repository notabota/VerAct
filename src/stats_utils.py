import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class StatResult:
    method_a: str
    method_b: str
    metric: str
    mean_a: float
    mean_b: float
    ci_a: Tuple[float, float]
    ci_b: Tuple[float, float]
    difference: float
    p_value: float
    significant: bool
    effect_size: float
    effect_interpretation: str


def wilson_ci(k: int, n: int, conf: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt((p * (1-p) + z**2 / (4*n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar(df: pd.DataFrame, method_a: str, method_b: str,
            success: str = "SUCCESS") -> Tuple[float, float]:
    a = df[df['method'] == method_a].set_index(['domain', 'difficulty', 'seed'])['status']
    b = df[df['method'] == method_b].set_index(['domain', 'difficulty', 'seed'])['status']
    common = a.index.intersection(b.index)
    a_ok = (a.loc[common] == success).values
    b_ok = (b.loc[common] == success).values
    n01 = np.sum(~a_ok & b_ok)
    n10 = np.sum(a_ok & ~b_ok)
    if n01 + n10 == 0:
        return 0.0, 1.0
    chi2 = (abs(n01 - n10) - 1)**2 / (n01 + n10)
    return chi2, 1 - stats.chi2.cdf(chi2, df=1)


def cohens_h(p1: float, p2: float) -> float:
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def interpret_h(h: float) -> str:
    h = abs(h)
    if h < 0.2:
        return "negligible"
    elif h < 0.5:
        return "small"
    elif h < 0.8:
        return "medium"
    return "large"


def bootstrap_ci(data: np.ndarray, stat=np.mean, n_boot: int = 10000,
                 conf: float = 0.95) -> Tuple[float, float]:
    if len(data) == 0:
        return (np.nan, np.nan)
    samples = [stat(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    alpha = 1 - conf
    return (np.percentile(samples, 100 * alpha / 2), np.percentile(samples, 100 * (1 - alpha / 2)))


def compare_methods(df: pd.DataFrame, method_a: str, method_b: str,
                    metric: str = "success_rate") -> StatResult:
    a_data = df[df['method'] == method_a]
    b_data = df[df['method'] == method_b]

    if metric == "success_rate":
        a_k = (a_data['status'] == 'SUCCESS').sum()
        b_k = (b_data['status'] == 'SUCCESS').sum()
    else:
        a_k = (a_data['status'] == 'VIOLATION').sum()
        b_k = (b_data['status'] == 'VIOLATION').sum()

    n_a, n_b = len(a_data), len(b_data)
    rate_a = a_k / n_a if n_a > 0 else 0
    rate_b = b_k / n_b if n_b > 0 else 0
    ci_a = wilson_ci(a_k, n_a)
    ci_b = wilson_ci(b_k, n_b)
    _, p = mcnemar(df, method_a, method_b, "SUCCESS" if metric == "success_rate" else "VIOLATION")
    h = cohens_h(rate_a, rate_b)

    return StatResult(
        method_a=method_a, method_b=method_b, metric=metric,
        mean_a=rate_a * 100, mean_b=rate_b * 100,
        ci_a=(ci_a[0] * 100, ci_a[1] * 100), ci_b=(ci_b[0] * 100, ci_b[1] * 100),
        difference=(rate_a - rate_b) * 100, p_value=p,
        significant=p < 0.05, effect_size=h, effect_interpretation=interpret_h(h)
    )


def compute_all_comparisons(df: pd.DataFrame, baseline: str = "ReAct",
                            methods: List[str] = None) -> pd.DataFrame:
    if methods is None:
        methods = df['method'].unique().tolist()
    rows = []
    for m in methods:
        if m == baseline:
            continue
        for metric in ["success_rate", "violation_rate"]:
            r = compare_methods(df, m, baseline, metric)
            rows.append({
                'method': m, 'baseline': baseline, 'metric': metric,
                'method_rate': r.mean_a, 'baseline_rate': r.mean_b,
                'method_ci': f"[{r.ci_a[0]:.1f}, {r.ci_a[1]:.1f}]",
                'difference': r.difference, 'p_value': r.p_value,
                'significant': r.significant, 'effect_size': r.effect_size,
                'effect': r.effect_interpretation
            })
    return pd.DataFrame(rows)

# Initial test to prove VerAct works before continue
def generate_significance_table(df: pd.DataFrame) -> str:
    comp = compute_all_comparisons(df, baseline="ReAct")
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Statistical Comparison (vs ReAct baseline)}",
        r"\label{tab:significance}",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"Method & Success (\%) & $\Delta$ & $p$-value & Effect \\", r"\midrule"
    ]
    for _, row in comp[comp['metric'] == 'success_rate'].iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        lines.append(f"{row['method']} & {row['method_rate']:.1f} {row['method_ci']} & "
                     f"{row['difference']:+.1f}{sig} & {row['p_value']:.3f} & {row['effect']} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\multicolumn{5}{l}{\footnotesize * $p<0.05$, ** $p<0.01$, *** $p<0.001$} \\",
        r"\end{tabular}", r"\end{table}"
    ])
    return "\n".join(lines)


def print_statistical_summary(df: pd.DataFrame):
    print("\nStatistical analysis")
    comp = compute_all_comparisons(df, baseline="ReAct")
    print("\nSuccess rate comparisons (vs ReAct):")
    for _, row in comp[comp['metric'] == 'success_rate'].iterrows():
        sig = "**" if row['significant'] else ""
        print(f"  {row['method']:20s}: {row['method_rate']:5.1f}% "
              f"(d={row['difference']:+5.1f}%, p={row['p_value']:.3f}{sig})")
