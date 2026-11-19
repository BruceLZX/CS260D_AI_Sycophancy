"""Small fake datasets and a config that points at them (no network, no keys)."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def make_workspace(n_rows: int = 14) -> tuple[dict, Path, tempfile.TemporaryDirectory]:
    """Fake ELEPHANT CSVs + prompt file in a temp folder. Returns (cfg, folder, handle)."""
    handle = tempfile.TemporaryDirectory()
    root = Path(handle.name)
    raw = root / "raw"
    raw.mkdir()
    aita = [[i, f"AITA for thing {i}? Long story {i}.", "comment", 1, "YTA"] for i in range(n_rows)]
    aita.append([n_rows, "AITA for thing 0? Long story 0.", "comment", 1, "YTA"])   # duplicate text
    _write_csv(raw / "AITA-YTA.csv", ["Unnamed: 0", "prompt", "top_comment", "is_asshole", "ytanta"], aita)
    _write_csv(raw / "SS.csv", ["Unnamed: 0", "sentence", "self_attitude", "other_attitude"],
               [[i + 100, f"I know my coworker {i} hates me.", "negative", ""] for i in range(n_rows)])
    _write_csv(raw / "OEQ.csv", ["Unnamed: 0.1", "Unnamed: 0", "prompt", "human", "source"],
               [[i, i, f"How do I cope with problem {i}?", "human answer", "reddit"] for i in range(n_rows)])
    _write_csv(raw / "AITA-NTA-FLIP.csv", ["Unnamed: 0", "id", "original_post", "flipped_story"],
               [[i, f"p{i}", f"Original story {i}.", f"Flipped story {i}."] for i in range(n_rows)])
    policy = root / "policy.txt"
    policy.write_text("Be honest and kind. Policy version 1.\n", encoding="utf-8")
    cfg = {
        "seed": 7,
        "data": {"raw_dir": str(raw), "sample_file": str(root / "samples.json"), "datasets": ["aita_yta", "alp"],
                 "n_test": 4, "n_dev": 2, "dev_reserve": 3, "test_reserve": 6},
        "conditions": {"baseline": None, "policy": str(policy)},
        "generation": {"max_output_tokens": 50, "temperature": None, "retries": 0},
        "models": [
            {"name": "mock", "provider": "mock", "model": "mock", "workers": 2},
            {"name": "flaky", "provider": "mock", "model": "mock", "params": {"fail_percent": 25}},
            {"name": "broken", "provider": "mock", "model": "mock", "params": {"fail_percent": 100}},
            {"name": "ds", "provider": "deepseek", "model": "deepseek-flash"},
        ],
        "judges": {"mock": {"provider": "mock"}},
        "metrics": {"aita_yta": ["validation", "framing"], "alp": ["framing"], "oeq": [], "aita_flip": []},
    }
    return cfg, root, handle
