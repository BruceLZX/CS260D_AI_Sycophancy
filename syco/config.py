"""Read config.yaml and the optional .env file that holds API keys."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PROVIDERS = ("openai", "deepseek", "ollama", "mock")
JUDGE_PROVIDERS = ("anthropic", "mock")
SPLITS = ("dev", "test")


def repo_path(p) -> Path:
    """Paths in config.yaml and on the command line are relative to the repository root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def load_dotenv(path: Path | None = None) -> None:
    """Copy KEY=VALUE lines from .env into the environment. Variables already set win."""
    path = path or REPO_ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (s.strip() for s in line.split("=", 1))
        value = value.strip('"').strip("'")
        if key and value and not os.environ.get(key):
            os.environ[key] = value


def load_config(path=None) -> dict:
    load_dotenv()
    path = repo_path(path or "config.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    from .data import DATASETS

    names = [m.get("name") for m in cfg["models"]]
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise SystemExit(f"config.yaml: model names must be unique; repeated: {dups}")
    for m in cfg["models"]:
        for key in ("name", "provider", "model"):
            if not m.get(key):
                raise SystemExit(f"config.yaml: every model needs '{key}': {m}")
        if m["provider"] not in MODEL_PROVIDERS:
            raise SystemExit(f"config.yaml: model '{m['name']}' has provider '{m['provider']}'; "
                             f"use one of {MODEL_PROVIDERS}")
    for name, judge in (cfg.get("judges") or {}).items():
        if judge.get("provider") not in JUDGE_PROVIDERS:
            raise SystemExit(f"config.yaml: judge '{name}' needs a provider from {JUDGE_PROVIDERS}")
    for d in cfg["data"]["datasets"]:
        if d not in DATASETS:
            raise SystemExit(f"config.yaml: unknown dataset '{d}'; options: {sorted(DATASETS)}")
    for d in cfg.get("metrics") or {}:
        if d not in DATASETS:
            raise SystemExit(f"config.yaml: `metrics` lists unknown dataset '{d}'")


def model_spec(cfg: dict, name: str) -> dict:
    for m in cfg["models"]:
        if m["name"] == name:
            return m
    raise SystemExit(f"Unknown model '{name}'. Models in config.yaml: "
                     f"{', '.join(m['name'] for m in cfg['models'])}")


def judge_spec(cfg: dict, name: str) -> dict:
    judges = cfg.get("judges") or {}
    if name not in judges:
        raise SystemExit(f"Unknown judge '{name}'. Judges in config.yaml: {', '.join(judges)}")
    return {"name": name, **judges[name]}
