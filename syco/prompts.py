"""System prompts for the experimental conditions (the policy prompt itself is the team's;
this file only loads it)."""
from __future__ import annotations

from dataclasses import dataclass

from .config import repo_path
from .io import sha12

PLACEHOLDER = "<<POLICY PROMPT NOT WRITTEN YET>>"


@dataclass(frozen=True)
class Condition:
    name: str
    system: str | None     # the system message; None = no system message
    source: str | None     # file it was read from
    sha: str               # fingerprint of the exact text sent ("none" without a system message)

    @property
    def is_placeholder(self) -> bool:
        return bool(self.system) and PLACEHOLDER in self.system

    @property
    def words(self) -> int:
        return len((self.system or "").split())


def load_conditions(cfg: dict, names=None) -> list[Condition]:
    """Conditions from config.yaml, in config order unless `names` is given."""
    table = cfg["conditions"]
    out = []
    for name in names or list(table):
        if name not in table:
            raise SystemExit(f"Unknown condition '{name}'. Conditions in config.yaml: {', '.join(table)}")
        source = table[name]
        if source is None:
            out.append(Condition(name, None, None, "none"))
            continue
        path = repo_path(source)
        if not path.exists():
            raise SystemExit(f"Condition '{name}': prompt file {path} not found.")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise SystemExit(f"Condition '{name}': {path} is empty.")
        out.append(Condition(name, text, str(source), sha12(text)))
    return out
