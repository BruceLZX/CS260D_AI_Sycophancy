"""Run models x conditions x items and save every response.

Output: outputs/<split>/generations/<model>.jsonl, one JSON record per response (one file per
model, so teammates never edit the same file). Running the same command again resumes:
finished calls are skipped and failed ones are retried.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import __version__
from .config import model_spec, repo_path
from .data import load_items
from .io import append_jsonl, read_jsonl, utc_now
from .models import ModelError, call_with_retries, get_model
from .prompts import load_conditions

ASSUMED_OUTPUT_TOKENS = 600     # used only for --dry-run cost estimates
MAX_CONSECUTIVE_ERRORS = 10     # stop a run that is clearly broken (bad key, wrong model ID, ...)


def generations_path(out_root, split: str, model_name: str) -> Path:
    return Path(out_root) / split / "generations" / f"{model_name}.jsonl"


def record_key(rec: dict, split: str) -> tuple:
    """Identity of one call. Dev runs also key on the prompt version, so each prompt draft gets
    its own responses; test runs must all use one frozen prompt."""
    key = (rec["dataset"], rec["item_id"], rec["condition"])
    return key + (rec["system_prompt_sha"],) if split == "dev" else key


def latest_ok(records: list[dict], split: str) -> dict:
    """key -> the most recent successful record for that key."""
    out = {}
    for rec in records:
        if not rec.get("error"):
            out[record_key(rec, split)] = rec
    return out


def check_prompt_versions(path: Path, records: list[dict], conditions, split: str) -> None:
    """Refuse to mix prompt versions in a test file."""
    if split != "test":
        return
    for cond in conditions:
        old = {r["system_prompt_sha"] for r in records if not r.get("error") and r["condition"] == cond.name}
        if old and old != {cond.sha}:
            raise SystemExit(
                f"{path} already has '{cond.name}' responses made with a different system prompt "
                f"({', '.join(sorted(old))}) than {cond.source or 'none'} has now ({cond.sha}).\n"
                "All test responses must use the same frozen prompt. Restore the prompt file, or move "
                f"{path.name} into outputs/old/ and run again to redo this model.")


def estimate(jobs, spec: dict) -> str:
    n_in = sum((len(cond.system or "") + len(item.text)) / 4 for item, cond in jobs)
    n_out = len(jobs) * ASSUMED_OUTPUT_TOKENS
    msg = (f"{len(jobs)} calls, about {n_in / 1e6:.2f}M input and {n_out / 1e6:.2f}M output tokens "
           f"(assuming {ASSUMED_OUTPUT_TOKENS} output tokens per reply)")
    price = spec.get("price_per_million")
    if price:
        usd = n_in / 1e6 * price["input"] + n_out / 1e6 * price["output"]
        msg += f", about ${usd:.2f} at the prices in config.yaml"
    return msg


def run(cfg: dict, *, models, split: str = "test", conditions=None, datasets=None, n=None,
        out_root="outputs", workers=None, dry_run: bool = False) -> dict:
    out_root = repo_path(out_root)
    conds = load_conditions(cfg, conditions)
    items = load_items(cfg, split, datasets, n)
    specs = [model_spec(cfg, m) for m in models]
    placeholders = [c.source for c in conds if c.is_placeholder]
    if placeholders and dry_run:
        print(f"note: {', '.join(placeholders)} is still the placeholder, so the estimate leaves out the "
              "real prompt's length (each 500-word prompt adds about 650 input tokens per call).")
    elif placeholders and any(s["provider"] != "mock" for s in specs):
        raise SystemExit(f"{', '.join(placeholders)} still holds the placeholder text. Real models run only "
                         "after the policy prompt is written (prompts/README.md). The mock model runs without it.")
    summary = {}
    for spec in specs:
        path = generations_path(out_root, split, spec["name"])
        records = read_jsonl(path)
        check_prompt_versions(path, records, conds, split)
        done = latest_ok(records, split)
        jobs = [(item, cond) for item in items for cond in conds
                if record_key({"dataset": item.dataset, "item_id": item.item_id, "condition": cond.name,
                               "system_prompt_sha": cond.sha}, split) not in done]
        total = len(items) * len(conds)
        print(f"[{spec['name']}] {len(items)} items x {len(conds)} conditions: {len(jobs)} calls to make, "
              f"{total - len(jobs)} already done -> {path}")
        if not jobs:
            summary[spec["name"]] = {"calls": 0, "errors": 0}
            continue
        if dry_run:
            print(f"[{spec['name']}] dry run: {estimate(jobs, spec)}")
            summary[spec["name"]] = {"calls": len(jobs), "dry_run": True}
            continue
        model = get_model(spec, cfg["generation"])
        failed_before = {record_key(r, split) for r in records if r.get("error")}
        summary[spec["name"]] = _execute(model, spec, jobs, split, path, cfg, workers,
                                         fresh_model=not done, failed_before=failed_before)
    return summary


def _execute(model, spec, jobs, split, path, cfg, workers, fresh_model=True, failed_before=frozenset()) -> dict:
    """fresh_model: nothing has worked for this model yet, so probe a few calls first.
    failed_before: keys that failed in an earlier run; failing again does not count toward the
    stop-after-10-failures-in-a-row rule (some items may fail every time, e.g. content filters)."""
    retries = int(cfg["generation"].get("retries", 6))
    workers = int(workers or spec.get("workers") or 1)
    settings = model.settings()

    def one(item, cond) -> dict:
        rec = {"dataset": item.dataset, "item_id": item.item_id, "split": split, "model": spec["name"],
               "condition": cond.name, "system_prompt_sha": cond.sha, "response": "", "error": None}
        try:
            gen = call_with_retries(lambda: model.generate(cond.system, item.text), retries=retries)
            rec.update(response=gen.text, finish_reason=gen.finish_reason, input_tokens=gen.input_tokens,
                       output_tokens=gen.output_tokens, reasoning_tokens=gen.reasoning_tokens,
                       served_model=gen.served_model, latency_s=gen.latency_s)
            if gen.extra:
                rec["extra"] = gen.extra
            if not gen.text:
                rec["error"] = f"empty response (finish_reason={gen.finish_reason})"
        except ModelError as e:
            rec["error"] = str(e)[:500]
        rec.update(settings=settings, created_at=utc_now(), framework=__version__)
        append_jsonl(path, rec)
        return rec

    # For a model with no successful call yet, run a few calls one at a time first, so a wrong
    # key, model ID or parameter fails fast instead of failing hundreds of times.
    probes, probe_errors = 0, []
    if fresh_model:
        while probes < min(3, len(jobs)):
            rec = one(*jobs[probes])
            probes += 1
            if not rec["error"]:
                break
            probe_errors.append(rec["error"])
        else:
            raise SystemExit(f"[{spec['name']}] the first {probes} calls failed. Last error: {probe_errors[-1]}\n"
                             "Fix the problem (try `python -m syco smoke`) and run again.")
    done, errors, streak, start = probes, len(probe_errors), 0, time.time()
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(one, item, cond) for item, cond in jobs[probes:]]
    try:
        for fut in as_completed(futures):
            rec = fut.result()
            done += 1
            if rec["error"]:
                errors += 1
                if record_key(rec, split) not in failed_before:
                    streak += 1
                if errors <= 5:
                    print(f"  error on {rec['item_id']} / {rec['condition']}: {rec['error'][:200]}")
                if streak >= MAX_CONSECUTIVE_ERRORS:
                    raise SystemExit(f"[{spec['name']}] {streak} failures in a row; stopping. Last error: "
                                     f"{rec['error'][:300]}\nFix the problem and run again to resume.")
            else:
                streak = 0
            if done % 20 == 0 or done == len(jobs):
                rate = done / max(time.time() - start, 1e-9)
                left = (len(jobs) - done) / rate if rate else 0
                print(f"[{spec['name']}] {done}/{len(jobs)} done, {errors} errors, {rate:.2f} calls/s, "
                      f"about {left / 60:.0f} min left", flush=True)
    except (KeyboardInterrupt, SystemExit):
        print("Stopping. Finished calls are saved; run the same command again to resume.")
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    if errors:
        print(f"[{spec['name']}] {errors} calls failed; run the same command again to retry them.")
    return {"calls": done, "errors": errors}
