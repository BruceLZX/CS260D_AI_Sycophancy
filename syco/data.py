"""ELEPHANT datasets: reading the raw CSVs, the fixed item sample, and items.

An *item* is one user message. Every model answers every item under every condition.
Item IDs are the same on every machine: "<dataset>-<row id>". AITA-FLIP has two items per
conflict: "aita_flip-<post id>-og" (original post) and "aita_flip-<post id>-flip".
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from .config import repo_path
from .io import file_sha256, utc_now

DEV_RESERVE = 50     # first 50 shuffled units per dataset: prompt development only
TEST_RESERVE = 500   # next 500: the test list; a run uses its first n_test units

DATASETS = {
    "aita_yta": {
        "file": "AITA-YTA.csv", "id_col": "Unnamed: 0", "text_col": "prompt",
        "description": "r/AmITheAsshole posts the community judged YTA (the poster is at fault).",
    },
    "alp": {
        "file": "SS.csv", "id_col": "Unnamed: 0", "text_col": "sentence",
        "description": "Assumption-laden statements (ALP; ELEPHANT's file is named SS.csv).",
    },
    "oeq": {
        "file": "OEQ.csv", "id_col": "Unnamed: 0", "text_col": "prompt",
        "description": "Open-ended personal-advice questions (emotional-support cases).",
    },
    "aita_flip": {
        "file": "AITA-NTA-FLIP.csv", "id_col": "id", "text_col": None,
        "description": "Conflict pairs: the original post (poster not at fault) and the same "
                       "conflict told by the other party.",
    },
}


@dataclass(frozen=True)
class Item:
    dataset: str
    item_id: str
    text: str                       # the user message, sent exactly as is
    meta: dict = field(default_factory=dict, compare=False, hash=False)


def raw_path(cfg: dict, name: str):
    return repo_path(cfg["data"]["raw_dir"]) / DATASETS[name]["file"]


def _units(cfg: dict, name: str) -> dict[str, list[Item]]:
    """Sampling units of one dataset -> their items. A unit is one row, except in AITA-FLIP
    where it is one pair (two items). Rows whose text repeats an earlier row are dropped."""
    import pandas as pd

    path = raw_path(cfg, name)
    if not path.exists():
        raise SystemExit(f"Missing {path}\nRun `python -m syco fetch-data` first (README, Setup).")
    spec = DATASETS[name]
    seen: set[str] = set()
    units: dict[str, list[Item]] = {}
    for row in pd.read_csv(path).to_dict("records"):
        uid = f"{name}-{row[spec['id_col']]}"
        if name == "aita_flip":
            texts = {"og": str(row["original_post"]).strip(), "flip": str(row["flipped_story"]).strip()}
            if any(t in seen for t in texts.values()):
                continue
            seen.update(texts.values())
            units[uid] = [Item(name, f"{uid}-{side}", text, {"pair": uid, "side": side})
                          for side, text in texts.items()]
        else:
            text = str(row[spec["text_col"]]).strip()
            if text in seen:
                continue
            seen.add(text)
            units[uid] = [Item(name, uid, text)]
    return units


def build_samples(cfg: dict) -> dict:
    """Shuffle every dataset once with the config seed and keep the unit IDs.
    data/samples.json is committed, so all teammates use exactly the same items."""
    seed = cfg["seed"]
    dev_n = int(cfg["data"].get("dev_reserve", DEV_RESERVE))
    test_n = int(cfg["data"].get("test_reserve", TEST_RESERVE))
    out = {"seed": seed, "created_at": utc_now(), "dev_reserve": dev_n, "test_reserve": test_n,
           "note": "Unit IDs only (no text). dev = prompt development; test = the reported results. "
                   "A run with n items per dataset uses the first n IDs of the list.",
           "datasets": {}}
    for name in DATASETS:
        ids = sorted(_units(cfg, name))
        random.Random(f"{seed}:{name}").shuffle(ids)
        out["datasets"][name] = {
            "source_file": DATASETS[name]["file"],
            "source_sha256": file_sha256(raw_path(cfg, name)),
            "units_available": len(ids),
            "dev": ids[:dev_n],
            "test": ids[dev_n:dev_n + test_n],
        }
    return out


def write_samples(cfg: dict, force: bool = False):
    path = repo_path(cfg["data"]["sample_file"])
    if path.exists() and not force:
        raise SystemExit(f"{path} already exists and fixes everyone's items. "
                         "Rebuild only if the team agrees (--force).")
    samples = build_samples(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(samples, indent=1) + "\n", encoding="utf-8")
    return path


def load_samples(cfg: dict) -> dict:
    path = repo_path(cfg["data"]["sample_file"])
    if not path.exists():
        raise SystemExit(f"Missing {path}. It should come with the repository (git pull).")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_source(cfg: dict, name: str, entry: dict) -> None:
    digest = file_sha256(raw_path(cfg, name))
    if digest != entry["source_sha256"]:
        print(f"warning: {raw_path(cfg, name)} differs from the file the sample was built from. "
              "Re-run `python -m syco fetch-data --force` so everyone uses the same data.")


def load_items(cfg: dict, split: str, datasets=None, n: int | None = None) -> list[Item]:
    """Items of `split` for the given datasets: the first n units of each list
    (n defaults to data.n_test or data.n_dev in config.yaml; AITA-FLIP units are pairs)."""
    if split not in ("dev", "test"):
        raise SystemExit(f"split must be dev or test, not {split!r}")
    samples = load_samples(cfg)
    n = int(n or cfg["data"]["n_test" if split == "test" else "n_dev"])
    items: list[Item] = []
    for name in datasets or cfg["data"]["datasets"]:
        if name not in DATASETS:
            raise SystemExit(f"Unknown dataset '{name}'; options: {sorted(DATASETS)}")
        entry = samples["datasets"][name]
        ids = entry[split]
        if n > len(ids):
            raise SystemExit(f"{name}: asked for {n} {split} units but the sample has {len(ids)}.")
        _check_source(cfg, name, entry)
        units = _units(cfg, name)
        for uid in ids[:n]:
            if uid not in units:
                raise SystemExit(f"{uid} is not in {raw_path(cfg, name)}; re-run `python -m syco fetch-data --force`.")
            items.extend(units[uid])
    return items


def item_lookup(cfg: dict, split: str, datasets) -> dict[tuple[str, str], Item]:
    """(dataset, item_id) -> Item for every sampled unit of the split (used by the judge)."""
    samples = load_samples(cfg)
    lookup = {}
    for name in datasets:
        units = _units(cfg, name)
        for uid in samples["datasets"][name][split]:
            for item in units.get(uid, []):
                lookup[(item.dataset, item.item_id)] = item
    return lookup
