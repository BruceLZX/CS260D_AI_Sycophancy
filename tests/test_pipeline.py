"""End-to-end checks with fake data, the mock model and the mock judge (no network, no keys)."""
from __future__ import annotations

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from syco import analyze, generate, status
from syco.config import REPO_ROOT, load_config
from syco.data import build_samples, load_items, write_samples
from syco.generate import generations_path
from syco.io import append_jsonl, read_jsonl
from syco.judge import run as judge_run
from syco.prompts import PLACEHOLDER
from tests.helpers import make_workspace


def quiet(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()) as buf:
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.cfg, self.root, self._handle = make_workspace()
        write_samples(self.cfg)
        self.out = self.root / "outputs"

    def tearDown(self):
        self._handle.cleanup()

    def gen(self, **kwargs):
        kwargs.setdefault("models", ["mock"])
        return quiet(generate.run, self.cfg, out_root=self.out, **kwargs)[0]

    # ---- data ----
    def test_sample_is_deterministic_and_seed_dependent(self):
        a, b = build_samples(self.cfg), build_samples(self.cfg)
        for name in a["datasets"]:
            self.assertEqual(a["datasets"][name]["test"], b["datasets"][name]["test"])
        other = dict(self.cfg, seed=8)
        self.assertNotEqual(a["datasets"]["alp"]["test"], build_samples(other)["datasets"]["alp"]["test"])

    def test_duplicate_texts_are_dropped_and_splits_disjoint(self):
        entry = build_samples(self.cfg)["datasets"]["aita_yta"]
        self.assertEqual(entry["units_available"], 14)            # 15 rows, one repeats row 0
        self.assertFalse(set(entry["dev"]) & set(entry["test"]))

    def test_flip_units_are_pairs(self):
        items = load_items(self.cfg, "test", ["aita_flip"], 2)
        self.assertEqual(len(items), 4)
        self.assertEqual({i.meta["side"] for i in items}, {"og", "flip"})
        self.assertTrue(all(i.item_id.endswith(("-og", "-flip")) for i in items))

    def test_refuses_to_overwrite_sample(self):
        with self.assertRaises(SystemExit):
            write_samples(self.cfg)

    # ---- generation ----
    def test_generate_resumes_and_guards_prompt_version(self):
        self.assertEqual(self.gen()["mock"]["calls"], 16)          # 2 datasets x 4 items x 2 conditions
        self.assertEqual(self.gen()["mock"]["calls"], 0)           # resume: nothing left
        Path(self.cfg["conditions"]["policy"]).write_text("Policy version 2.", encoding="utf-8")
        with self.assertRaises(SystemExit) as ctx:
            self.gen()
        self.assertIn("different system prompt", str(ctx.exception))

    def test_dev_split_keeps_one_set_per_prompt_draft(self):
        self.assertEqual(self.gen(split="dev")["mock"]["calls"], 8)    # 2 datasets x 2 items x 2 conditions
        Path(self.cfg["conditions"]["policy"]).write_text("Policy draft 2.", encoding="utf-8")
        self.assertEqual(self.gen(split="dev")["mock"]["calls"], 4)    # only the policy condition again

    def test_placeholder_blocks_real_models_but_not_mock_or_dry_run(self):
        Path(self.cfg["conditions"]["policy"]).write_text(PLACEHOLDER + "\nfill me", encoding="utf-8")
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            with self.assertRaises(SystemExit) as ctx:
                self.gen(models=["ds"])
            self.assertIn("placeholder", str(ctx.exception))
            self.assertTrue(self.gen(models=["ds"], dry_run=True)["ds"]["dry_run"])
        self.assertEqual(self.gen(models=["mock"])["mock"]["calls"], 16)

    def test_failures_are_saved_and_retried(self):
        first = self.gen(models=["flaky"])["flaky"]
        self.assertGreater(first["errors"], 0)
        path = generations_path(self.out, "test", "flaky")
        errors = [r for r in read_jsonl(path) if r["error"]]
        self.assertEqual(len(errors), first["errors"])
        self.assertEqual(self.gen(models=["flaky"])["flaky"]["calls"], first["errors"])   # only failed ones again
        _, report = quiet(status.run, self.cfg, out_root=self.out)
        self.assertIn("flaky", report)

    def test_broken_model_stops_after_first_calls(self):
        with self.assertRaises(SystemExit) as ctx:
            self.gen(models=["broken"])
        self.assertIn("first 3 calls failed", str(ctx.exception))
        self.assertEqual(len(read_jsonl(generations_path(self.out, "test", "broken"))), 3)

    def test_half_written_line_is_skipped_and_repaired(self):
        self.gen()
        path = generations_path(self.out, "test", "mock")
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"dataset": "alp", "item_')                    # an interrupted write
        records, _ = quiet(read_jsonl, path)
        self.assertEqual(len(records), 16)
        append_jsonl(path, {"x": 1})
        records, _ = quiet(read_jsonl, path)
        self.assertEqual(records[-1], {"x": 1})

    # ---- judge + analyze ----
    def test_judge_and_analyze(self):
        self.gen()
        res, _ = quiet(judge_run.run, self.cfg, models=["mock"], judge="mock", out_root=self.out)
        self.assertEqual(res["mock"]["calls"], 24)                 # aita: 4x2x2 metrics, alp: 4x2x1
        res, _ = quiet(judge_run.run, self.cfg, models=["mock"], judge="mock", out_root=self.out)
        self.assertEqual(res["mock"]["calls"], 0)
        # a re-generated response is judged again and replaces the old judgment
        gen_path = generations_path(self.out, "test", "mock")
        old = next(r for r in read_jsonl(gen_path) if r["dataset"] == "alp" and r["condition"] == "baseline")
        append_jsonl(gen_path, dict(old, response="A different, critical reply."))
        res, _ = quiet(judge_run.run, self.cfg, models=["mock"], judge="mock", out_root=self.out)
        self.assertEqual(res["mock"]["calls"], 1)
        out_dir, _ = quiet(analyze.run, self.cfg, judge="mock", out_root=self.out)
        table = (out_dir / "summary.csv").read_text(encoding="utf-8")
        self.assertIn("diff_vs_baseline", table)
        df = analyze.current_judgments(self.out, "test", "mock")
        self.assertEqual(len(df), 24)                              # one current judgment per response x metric

    def test_unknown_metric_in_config_is_reported(self):
        self.gen()
        self.cfg["metrics"]["alp"] = ["framing", "policy_r9"]
        with self.assertRaises(SystemExit) as ctx:
            quiet(judge_run.run, self.cfg, models=["mock"], judge="mock", out_root=self.out)
        self.assertIn("policy_r9", str(ctx.exception))

    def test_mcnemar_exact(self):
        self.assertAlmostEqual(analyze.mcnemar_exact_p(0, 5), 0.0625)
        self.assertEqual(analyze.mcnemar_exact_p(3, 3), 1.0)
        self.assertEqual(analyze.mcnemar_exact_p(0, 0), 1.0)


class RepoConfigTest(unittest.TestCase):
    """The committed config.yaml and data/samples.json are valid."""

    def test_config_loads_and_groups_resolve(self):
        from syco.__main__ import resolve_models

        cfg = load_config()
        api = resolve_models(cfg, "api")
        self.assertTrue(api and all(cfg_m["provider"] in ("openai", "deepseek")
                                    for cfg_m in cfg["models"] if cfg_m["name"] in api))
        self.assertNotIn("mock", resolve_models(cfg, "all"))
        self.assertEqual(resolve_models(cfg, "mock,mock"), ["mock"])
        with self.assertRaises(SystemExit):
            resolve_models(cfg, "no-such-model")

    def test_sample_file_matches_local_data(self):
        cfg = load_config()
        raw = REPO_ROOT / cfg["data"]["raw_dir"]
        if not raw.exists():
            self.skipTest("raw data not downloaded (python -m syco fetch-data)")
        committed = json.loads((REPO_ROOT / cfg["data"]["sample_file"]).read_text(encoding="utf-8"))
        rebuilt = build_samples(cfg)
        for name, entry in committed["datasets"].items():
            self.assertEqual(entry["test"], rebuilt["datasets"][name]["test"], name)
            self.assertEqual(entry["dev"], rebuilt["datasets"][name]["dev"], name)
            self.assertEqual(entry["source_sha256"], rebuilt["datasets"][name]["source_sha256"], name)


if __name__ == "__main__":
    unittest.main()
