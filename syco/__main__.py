"""Command line: `python -m syco <command> [options]`. Run `python -m syco -h` for the list,
or `python -m syco <command> -h` for one command's options."""
from __future__ import annotations

import argparse
import textwrap

from .config import load_config, model_spec, repo_path

GROUPS = {"api": ("openai", "deepseek"), "local": ("ollama", "cecilia")}


def resolve_models(cfg: dict, value: str) -> list[str]:
    """'api', 'local', 'all' (every non-mock model) or a comma-separated list of names."""
    names: list[str] = []
    for part in (p.strip() for p in value.split(",")):
        if part in GROUPS:
            names += [m["name"] for m in cfg["models"] if m["provider"] in GROUPS[part]]
        elif part == "all":
            names += [m["name"] for m in cfg["models"] if m["provider"] != "mock"]
        elif part:
            model_spec(cfg, part)      # exits with the list of valid names if unknown
            names.append(part)
    return list(dict.fromkeys(names))


def _csv(value):
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def cmd_prompt(cfg, args):
    from .prompts import load_conditions

    for cond in load_conditions(cfg):
        if cond.system is None:
            print(f"{cond.name:<12} no system prompt")
            continue
        state = "PLACEHOLDER - not written yet" if cond.is_placeholder else f"{cond.words} words"
        print(f"{cond.name:<12} {cond.source}  fingerprint {cond.sha}  ({state})")
    print("The fingerprint is saved with every response; all models must show the same one.")


def cmd_show(cfg, args):
    from .data import DATASETS, load_items

    items = load_items(cfg, args.split, [args.dataset], args.n)
    print(f"{args.dataset}: {DATASETS[args.dataset]['description']}\n")
    for item in items:
        print(f"--- {item.item_id} ({len(item.text)} characters)")
        print(textwrap.shorten(item.text, args.chars, placeholder=" [...]") if args.chars else item.text)
        print()


def cmd_smoke(cfg, args):
    from .models import ModelError, get_model

    question = "In two sentences: how should I apologize to a friend I upset?"
    for name in resolve_models(cfg, args.models):
        spec = model_spec(cfg, name)
        print(f"[{name}] {spec['provider']} / {spec['model']} ...", flush=True)
        try:
            gen = get_model(spec, cfg["generation"]).generate(None, question)
        except (ModelError, SystemExit) as e:      # SystemExit: e.g. a missing API key
            print(f"  FAILED: {e}\n")
            continue
        print(f"  reply: {textwrap.shorten(gen.text, 300)}")
        print(f"  served model {gen.served_model}, {gen.input_tokens} in / {gen.output_tokens} out tokens"
              f"{f' ({gen.reasoning_tokens} reasoning)' if gen.reasoning_tokens else ''}, "
              f"{gen.latency_s}s, finish {gen.finish_reason}, extra {gen.extra or '-'}\n")


def cmd_compare(cfg, args):
    from .data import item_lookup
    from .generate import generations_path
    from .io import read_jsonl

    path = generations_path(repo_path(args.out), args.split, args.model)
    latest = {}
    for rec in read_jsonl(path):
        if not rec.get("error") and (not args.dataset or rec["dataset"] == args.dataset):
            latest[(rec["dataset"], rec["item_id"], rec["condition"])] = rec
    if not latest:
        raise SystemExit(f"No responses in {path}.")
    lookup = item_lookup(cfg, args.split, sorted({k[0] for k in latest}))
    keys = list(dict.fromkeys((k[0], k[1]) for k in latest))[: args.n]
    for dataset, item_id in keys:
        item = lookup.get((dataset, item_id))
        print("=" * 100)
        print(f"{item_id}\nUSER: {textwrap.shorten(item.text if item else '?', args.chars, placeholder=' [...]')}")
        for cond in cfg["conditions"]:
            rec = latest.get((dataset, item_id, cond))
            if rec:
                print(f"\n[{cond}] (prompt {rec['system_prompt_sha']})\n"
                      f"{textwrap.shorten(rec['response'], args.chars, placeholder=' [...]')}")
        print()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m syco", description="Sycophancy policy experiment (CS 260E).")
    parser.add_argument("--config", default="config.yaml", help="config file (default: config.yaml)")
    parser.add_argument("--model-config", help="append model specs from a separate YAML file")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    def add(name, help_text):
        p = sub.add_parser(name, help=help_text, description=help_text)
        return p

    def common(p, models=True, split=True, out=True):
        if models:
            p.add_argument("--models", required=True,
                           help="comma-separated names from config.yaml, or api / local / all / mock")
        if split:
            p.add_argument("--split", default="test", choices=["dev", "test"],
                           help="dev = prompt development, test = reported results (default)")
        if out:
            p.add_argument("--out", default="outputs", help="output folder (default: outputs)")

    p = add("fetch-data", "download the ELEPHANT datasets into data/raw/")
    p.add_argument("--force", action="store_true", help="download again")

    p = add("sample", "rebuild data/samples.json (the fixed item IDs; only if the team agrees)")
    p.add_argument("--force", action="store_true")

    p = add("prompt", "show each condition's system prompt file and fingerprint")

    p = add("show", "print items from a dataset")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--chars", type=int, default=1500, help="shorten texts to this many characters (0 = full)")

    p = add("smoke", "one short test call per model (checks key, model ID and parameters)")
    p.add_argument("--models", required=True)

    p = add("generate", "get model responses for items x conditions (resumable)")
    common(p)
    p.add_argument("--conditions", help="comma-separated (default: all in config.yaml)")
    p.add_argument("--datasets", help="comma-separated (default: data.datasets in config.yaml)")
    p.add_argument("--n", type=int, help="items per dataset (default: data.n_test or data.n_dev)")
    p.add_argument("--workers", type=int, help="parallel calls (default: the model's `workers`)")
    p.add_argument("--dry-run", action="store_true", help="count calls and estimate cost; call nothing")

    p = add("status", "progress of the generation runs")
    common(p, models=False)

    p = add("compare", "print one model's responses side by side across conditions")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--chars", type=int, default=1200)
    p.add_argument("--out", default="outputs")

    p = add("judge", "score saved responses with a judge (resumable)")
    common(p)
    p.add_argument("--judge", default="claude", help="judge name from config.yaml (default: claude)")
    p.add_argument("--metrics", help="comma-separated (default: `metrics` in config.yaml)")
    p.add_argument("--limit", type=int, help="at most this many judgments (for a quick test)")
    p.add_argument("--workers", type=int)
    p.add_argument("--dry-run", action="store_true")

    p = add("analyze", "summary tables (rates per condition and paired comparisons)")
    common(p, models=False)
    p.add_argument("--judge", default="claude")

    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.model_config)

    if args.command == "fetch-data":
        from .fetch_data import run
        run(cfg, force=args.force)
    elif args.command == "sample":
        from .data import write_samples
        print(f"Wrote {write_samples(cfg, force=args.force)}")
    elif args.command == "prompt":
        cmd_prompt(cfg, args)
    elif args.command == "show":
        cmd_show(cfg, args)
    elif args.command == "smoke":
        cmd_smoke(cfg, args)
    elif args.command == "generate":
        from .generate import run
        run(cfg, models=resolve_models(cfg, args.models), split=args.split, conditions=_csv(args.conditions),
            datasets=_csv(args.datasets), n=args.n, out_root=args.out, workers=args.workers, dry_run=args.dry_run)
    elif args.command == "status":
        from .status import run
        run(cfg, split=args.split, out_root=args.out)
    elif args.command == "compare":
        cmd_compare(cfg, args)
    elif args.command == "judge":
        from .judge.run import run
        run(cfg, models=resolve_models(cfg, args.models), judge=args.judge, split=args.split,
            metrics=_csv(args.metrics), out_root=args.out, workers=args.workers, limit=args.limit,
            dry_run=args.dry_run)
    elif args.command == "analyze":
        from .analyze import run
        run(cfg, split=args.split, judge=args.judge, out_root=args.out)


if __name__ == "__main__":
    main()
