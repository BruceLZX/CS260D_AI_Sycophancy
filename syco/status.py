"""Progress report for the generation runs: what is done, what failed, which prompt version."""
from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

from .config import repo_path
from .io import read_jsonl


def prompt_versions(out_root, split: str) -> dict:
    """condition -> {prompt fingerprint -> [models]} over all generation files of a split."""
    versions = defaultdict(lambda: defaultdict(set))
    for path in sorted((Path(out_root) / split / "generations").glob("*.jsonl")):
        for rec in read_jsonl(path):
            if not rec.get("error"):
                versions[rec["condition"]][rec["system_prompt_sha"]].add(path.stem)
    return {cond: {sha: sorted(models) for sha, models in shas.items()} for cond, shas in versions.items()}


def warn_mixed_prompts(out_root, split: str) -> bool:
    mixed = False
    for cond, shas in prompt_versions(out_root, split).items():
        if len(shas) > 1:
            mixed = True
            detail = "; ".join(f"{sha}: {', '.join(models)}" for sha, models in shas.items())
            print(f"WARNING: condition '{cond}' was run with {len(shas)} different prompt versions ({detail}). "
                  "Every model must use the same prompt; re-run the outdated models.")
    return mixed


def run(cfg: dict, *, split: str = "test", out_root="outputs") -> None:
    root = repo_path(out_root) / split / "generations"
    files = sorted(root.glob("*.jsonl"))
    if not files:
        print(f"No generation files in {root} yet.")
        return
    n_cfg = cfg["data"]["n_test" if split == "test" else "n_dev"]
    print(f"Split '{split}': config.yaml asks for {n_cfg} items per dataset (AITA-FLIP: pairs, so 2x items).")
    print(f"{'model':<22}{'dataset':<12}{'condition':<12}{'ok':>6}{'failed':>8}{'cut off':>9}"
          f"{'thinking':>10}{'median out tok':>16}")
    for path in files:
        ok, failed = {}, {}
        for rec in read_jsonl(path):
            key = (rec["dataset"], rec["item_id"], rec["condition"],
                   rec["system_prompt_sha"] if split == "dev" else "")
            (failed if rec.get("error") else ok)[key] = rec
        unresolved = {k: r for k, r in failed.items() if k not in ok}
        groups = defaultdict(lambda: {"ok": 0, "failed": 0, "cut": 0, "think": 0, "tokens": []})
        for (dataset, _, cond, _), rec in ok.items():
            g = groups[(dataset, cond)]
            g["ok"] += 1
            g["cut"] += rec.get("finish_reason") == "length"
            g["think"] += "<think>" in rec["response"] or bool((rec.get("extra") or {}).get("thinking_chars"))
            if rec.get("output_tokens"):
                g["tokens"].append(rec["output_tokens"])
        for (dataset, _, cond, _) in unresolved:
            groups[(dataset, cond)]["failed"] += 1
        for (dataset, cond), g in sorted(groups.items()):
            median = statistics.median(g["tokens"]) if g["tokens"] else "-"
            print(f"{path.stem:<22}{dataset:<12}{cond:<12}{g['ok']:>6}{g['failed']:>8}{g['cut']:>9}"
                  f"{g['think']:>10}{median:>16}")
        if unresolved:
            example = next(iter(unresolved.values()))
            print(f"  e.g. {example['item_id']} / {example['condition']}: {example['error'][:200]}")
    print("cut off = stopped at the token limit; thinking = reasoning text found (should be 0 when thinking is off).")
    if split == "test":
        warn_mixed_prompts(repo_path(out_root), split)
