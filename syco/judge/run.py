"""Score saved responses with a judge (owners: Alice + Yanzi).

Input:  outputs/<split>/generations/<model>.jsonl
Output: outputs/<split>/judgments/<judge>/<model>.jsonl, one record per response x metric.
Running again only judges what is new (new responses, or a rubric that was edited).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .. import __version__
from ..config import judge_spec, repo_path
from ..data import item_lookup
from ..generate import generations_path, latest_ok
from ..io import append_jsonl, read_jsonl, sha12, utc_now
from ..models.base import ModelError, call_with_retries
from .clients import get_judge
from .rubrics import RUBRICS


def judgments_path(out_root, split: str, judge: str, model_name: str) -> Path:
    return Path(out_root) / split / "judgments" / judge / f"{model_name}.jsonl"


def metrics_for(cfg: dict, dataset: str, only=None) -> list[str]:
    names = list((cfg.get("metrics") or {}).get(dataset) or [])
    if only:
        names = [m for m in names if m in only]
    unknown = [m for m in names if m not in RUBRICS]
    if unknown:
        raise SystemExit(f"config.yaml lists metric(s) {unknown} for {dataset}, but syco/judge/rubrics.py "
                         f"has no such rubric. Available: {', '.join(RUBRICS)}")
    return names


def run(cfg: dict, *, models, judge: str = "claude", split: str = "test", metrics=None,
        out_root="outputs", workers=None, limit=None, dry_run: bool = False) -> dict:
    out_root = repo_path(out_root)
    spec = judge_spec(cfg, judge)
    for m in metrics or []:
        if m not in RUBRICS:
            raise SystemExit(f"Unknown metric '{m}'. Available: {', '.join(RUBRICS)}")
    client = None
    summary = {}
    for model_name in models:
        gen_path = generations_path(out_root, split, model_name)
        gens = list(latest_ok(read_jsonl(gen_path), split).values())
        if not gens:
            print(f"[{model_name}] no responses in {gen_path}; run `generate` first.")
            continue
        lookup = item_lookup(cfg, split, sorted({g["dataset"] for g in gens}))
        out_path = judgments_path(out_root, split, judge, model_name)
        done = {(r["dataset"], r["item_id"], r["condition"], r["metric"], r["response_sha"], r["rubric_sha"])
                for r in read_jsonl(out_path) if not r.get("error")}
        jobs = []
        for gen in gens:
            for metric in metrics_for(cfg, gen["dataset"], metrics):
                rubric = RUBRICS[metric]
                key = (gen["dataset"], gen["item_id"], gen["condition"], metric, sha12(gen["response"]), rubric.sha)
                if key not in done:
                    jobs.append((gen, rubric))
        if limit:
            jobs = jobs[: int(limit)]
        print(f"[{model_name}] {len(jobs)} judgments to make with '{judge}' -> {out_path}")
        if not jobs:
            summary[model_name] = {"calls": 0, "errors": 0}
            continue
        if dry_run:
            chars = 0
            for g, rubric in jobs:
                item = lookup.get((g["dataset"], g["item_id"]))
                chars += len(rubric.build(item.text if item else "", g["response"]))
            msg = f"[{model_name}] dry run: {len(jobs)} calls, about {chars / 4 / 1e6:.2f}M input tokens"
            price = spec.get("price_per_million")
            if price:
                msg += f", about ${chars / 4 / 1e6 * price['input']:.2f}"
            print(msg)
            summary[model_name] = {"calls": len(jobs), "dry_run": True}
            continue
        client = client or get_judge(spec)
        summary[model_name] = _execute(client, spec, cfg, jobs, lookup, out_path, workers)
    return summary


def _execute(client, spec, cfg, jobs, lookup, out_path, workers) -> dict:
    retries = int(cfg["generation"].get("retries", 6))

    def one(gen: dict, rubric) -> dict:
        rec = {"dataset": gen["dataset"], "item_id": gen["item_id"], "model": gen["model"],
               "condition": gen["condition"], "metric": rubric.name, "label": None, "judge_reply": None,
               "judge": spec["name"], "judge_model": getattr(client, "model", None),
               "rubric_sha": rubric.sha, "response_sha": sha12(gen["response"]), "error": None}
        item = lookup.get((gen["dataset"], gen["item_id"]))
        if item is None:
            rec["error"] = "item text not found (data/samples.json or the raw data differs)"
        else:
            try:
                prompt = rubric.build(item.text, gen["response"])
                reply = call_with_retries(lambda: client.complete(rubric.system, prompt, rubric.max_tokens),
                                          retries=retries)
                rec.update(judge_reply=reply.text, label=rubric.parse(reply.text),
                           input_tokens=reply.input_tokens, output_tokens=reply.output_tokens)
                if rec["label"] is None:
                    rec["error"] = "could not read a label from the judge reply"
            except ModelError as e:
                rec["error"] = str(e)[:500]
        rec.update(created_at=utc_now(), framework=__version__)
        append_jsonl(out_path, rec)
        return rec

    done, errors, start = 0, 0, time.time()
    with ThreadPoolExecutor(max_workers=int(workers or spec.get("workers") or 4)) as pool:
        futures = [pool.submit(one, gen, rubric) for gen, rubric in jobs]
        try:
            for fut in as_completed(futures):
                rec = fut.result()
                done += 1
                if rec["error"]:
                    errors += 1
                    if errors <= 5:
                        print(f"  error on {rec['item_id']} / {rec['condition']} / {rec['metric']}: {rec['error'][:200]}")
                if done % 50 == 0 or done == len(jobs):
                    rate = done / max(time.time() - start, 1e-9)
                    print(f"  {done}/{len(jobs)} judged, {errors} errors, {rate:.1f}/s", flush=True)
        except KeyboardInterrupt:
            print("Stopping. Finished judgments are saved; run the same command again to resume.")
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    if errors:
        print(f"{errors} judgments failed; run the same command again to retry them.")
    return {"calls": done, "errors": errors}
