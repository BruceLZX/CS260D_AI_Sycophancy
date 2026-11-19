"""Summary tables from the judgments, for the slides and the paper.

For every dataset x metric x model:
- summary.csv: the rate under each condition;
- comparisons.csv: for each [condition, reference] pair in `analysis.comparisons` (config.yaml),
  the paired difference on the same items (condition minus reference) with a 95% bootstrap CI
  and, for 0/1 labels, an exact McNemar p-value. Without that setting, every condition is
  compared with the first one.
Output: outputs/<split>/tables/<judge>/summary.csv, comparisons.csv and summary.md
"""
from __future__ import annotations

import math
from pathlib import Path

from .config import repo_path
from .generate import generations_path
from .io import read_jsonl, sha12
from .status import warn_mixed_prompts


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar test on discordant pairs (b: 1 -> 0, c: 0 -> 1)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_ci(before, after, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of mean(after - before), resampling items (paired)."""
    import numpy as np

    diff = np.asarray(after, float) - np.asarray(before, float)
    if diff.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(reps, diff.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def comparison_pairs(cfg: dict) -> list[tuple[str, str]]:
    """[condition, reference] pairs from config.yaml, or every condition vs the first one."""
    conds = list(cfg["conditions"])
    pairs = (cfg.get("analysis") or {}).get("comparisons")
    if not pairs:
        return [(c, conds[0]) for c in conds[1:]]
    return [(cond, ref) for cond, ref in pairs]


def current_judgments(out_root, split: str, judge: str):
    """Judgments of the responses currently on file; a re-generated response replaces the old one."""
    import pandas as pd

    rows = []
    for path in sorted((Path(out_root) / split / "judgments" / judge).glob("*.jsonl")):
        current = {}
        for gen in read_jsonl(generations_path(out_root, split, path.stem)):
            if not gen.get("error"):
                current[(gen["dataset"], gen["item_id"], gen["condition"])] = sha12(gen["response"])
        latest = {}
        for rec in read_jsonl(path):
            if rec.get("error") or rec.get("label") is None:
                continue
            if current.get((rec["dataset"], rec["item_id"], rec["condition"])) != rec["response_sha"]:
                continue
            latest[(rec["dataset"], rec["item_id"], rec["condition"], rec["metric"])] = rec
        rows.extend(latest.values())
    return pd.DataFrame(rows)


def summarize(df, condition_order, pairs, reps: int = 2000):
    """Returns (rates, comparisons) as two DataFrames."""
    import pandas as pd

    rates, comparisons = [], []
    for (dataset, metric, model), group in df.groupby(["dataset", "metric", "model"], sort=True):
        wide = group.pivot_table(index="item_id", columns="condition", values="label", aggfunc="last")
        key = {"dataset": dataset, "metric": metric, "model": model}
        conds = [c for c in condition_order if c in wide.columns] + \
                [c for c in wide.columns if c not in condition_order]
        for cond in conds:
            col = wide[cond].dropna()
            rates.append({**key, "condition": cond, "n": int(col.size), "rate": float(col.mean())})
        for cond, ref in pairs:
            if cond not in wide.columns or ref not in wide.columns:
                continue
            pair = wide[[ref, cond]].dropna()
            low, high = bootstrap_ci(pair[ref], pair[cond], reps)
            row = {**key, "condition": cond, "reference": ref, "n_paired": int(len(pair)),
                   "diff": float(pair[cond].mean() - pair[ref].mean()), "ci_low": low, "ci_high": high,
                   "mcnemar_p": math.nan}
            if len(pair) and set(pair.to_numpy().ravel().tolist()) <= {0, 1}:
                b = int(((pair[ref] == 1) & (pair[cond] == 0)).sum())
                c = int(((pair[ref] == 0) & (pair[cond] == 1)).sum())
                row["mcnemar_p"] = mcnemar_exact_p(b, c)
            comparisons.append(row)
    return pd.DataFrame(rates), pd.DataFrame(comparisons)


def to_markdown(rates, comparisons) -> str:
    def fmt(value, spec):
        return "" if value is None or (isinstance(value, float) and math.isnan(value)) else format(value, spec)

    lines = ["Rate = share of responses labeled 1 (for ELEPHANT metrics, 1 is the sycophantic direction).", "",
             "| dataset | metric | model | condition | n | rate |", "|---|---|---|---|---:|---:|"]
    for r in rates.to_dict("records"):
        lines.append(f"| {r['dataset']} | {r['metric']} | {r['model']} | {r['condition']} | {r['n']} | "
                     f"{r['rate']:.3f} |")
    if len(comparisons):
        lines += ["", "Difference = condition minus reference on the same items, with a 95% bootstrap CI.", "",
                  "| dataset | metric | model | condition vs reference | n | difference [95% CI] | McNemar p |",
                  "|---|---|---|---|---:|---|---:|"]
        for r in comparisons.to_dict("records"):
            diff = ""
            if fmt(r["diff"], "+.3f"):
                diff = f"{r['diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
            lines.append(f"| {r['dataset']} | {r['metric']} | {r['model']} | {r['condition']} vs {r['reference']} | "
                         f"{r['n_paired']} | {diff} | {fmt(r['mcnemar_p'], '.3g')} |")
    return "\n".join(lines) + "\n"


def run(cfg: dict, *, split: str = "test", judge: str = "claude", out_root="outputs", reps: int = 2000) -> Path:
    out_root = repo_path(out_root)
    df = current_judgments(out_root, split, judge)
    if df.empty:
        raise SystemExit(f"No judgments in {out_root / split / 'judgments' / judge}; run `judge` first.")
    warn_mixed_prompts(out_root, split)
    rates, comparisons = summarize(df, list(cfg["conditions"]), comparison_pairs(cfg), reps)
    out_dir = out_root / split / "tables" / judge
    out_dir.mkdir(parents=True, exist_ok=True)
    rates.to_csv(out_dir / "summary.csv", index=False)
    comparisons.to_csv(out_dir / "comparisons.csv", index=False)
    md = to_markdown(rates, comparisons)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Saved summary.csv, comparisons.csv and summary.md in {out_dir}")
    return out_dir
