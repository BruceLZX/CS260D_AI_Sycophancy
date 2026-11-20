# Policy prompt vs. sycophancy — experiment code (CS 260E)

We test whether our policy framework, given to a model as a system prompt, changes how the model
answers personal-advice requests. Every model answers the same items under three system prompts:
a neutral baseline (`baseline`), the baseline plus a generic anti-sycophancy instruction
(`generic`), and the full policy prompt (`policy`). A Claude judge then scores the answers with
ELEPHANT's metrics (and our own rubric), and we compare the conditions on the same items.

```
data/samples.json (fixed items) ─┐
                                 ├─► generate ─► outputs/test/generations/<model>.jsonl
prompts/ (one per condition)     ┘                      │
                                              judge ◄───┘ ─► outputs/test/judgments/claude/<model>.jsonl
                                                                  │
                                              analyze ◄───────────┘ ─► outputs/test/tables/claude/summary.md
```

## Who does what (Nov 18 meeting)

| Part | Owner | Where in this repo | Due |
|---|---|---|---|
| Policy system prompt (2 others review) | Jing | `prompts/` | done 11/19 |
| API models: OpenAI, DeepSeek | Bruce | `syco/models/openai_api.py` → `outputs/test/generations/` | Sat 11/21 |
| Local models: Qwen, Llama (Ollama or Bolei) | Cecilia | `syco/models/` → `outputs/test/generations/` | Sat 11/21 |
| Evaluation: Claude judge, ELEPHANT metrics, policy-matrix rubric | Alice, Yanzi | `syco/judge/` → `outputs/test/judgments/`, `outputs/test/tables/` | Sun 11/22 |
| Slides | everyone | | draft Mon 11/23, final Tue 11/24 |

What already works: data download and the fixed sample, the three condition prompts, the OpenAI
and DeepSeek runner, the Ollama runner (tested only against a fake server so far), the Claude judge
with ELEPHANT's three metrics, the summary tables, and a mock model and mock judge that let you try
every step for free. What still needs you: the real runs and the policy-matrix rubric.

## Setup (everyone, about 5 minutes)

Python 3.10 or newer.

```bash
git clone https://github.com/BruceLZX/CS260D_AI_Sycophancy.git
cd CS260D_AI_Sycophancy
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m syco fetch-data            # ELEPHANT datasets (6.7 MB) into data/raw/
python -m unittest                   # about 30 tests, a few seconds, no keys needed
```

**API keys.** Copy `.env.example` to `.env` and fill in only the keys you use (OpenAI and DeepSeek
for Bruce, Anthropic for Alice and Yanzi). `.env` is ignored by git. Never commit a key or paste
one into the group chat. Setting the same variables in your shell also works.

**Try every step for free.** The mock model and mock judge need no key. Results go to
`outputs/_scratch/`, which git ignores:

```bash
python -m syco generate --models mock --n 5 --out outputs/_scratch
python -m syco judge    --models mock --judge mock --out outputs/_scratch
python -m syco analyze  --judge mock --out outputs/_scratch
python -m syco compare  --model mock --n 2 --out outputs/_scratch
```

`python -m syco -h` lists all commands; `python -m syco <command> -h` shows a command's options.

## Jing: the prompts (done 11/19)

The three condition prompts are in `prompts/`, copied word for word from the "Final project -
System prompt construction" tab of our plan doc, except for one missing space we added in the
policy prompt ("evidence. Support"). Each file is sent word for word as the system message to
every model.

| condition | file | prompt |
|---|---|---|
| `baseline` (A) | `prompts/baseline_system_prompt.txt` | neutral baseline (24 words) |
| `generic` (B) | `prompts/generic_system_prompt.txt` | the baseline plus a generic anti-sycophancy instruction (44 words) |
| `policy` (C) | `prompts/policy_system_prompt.txt` | the full policy prompt (599 words) |

After the test runs start, the prompts must not change. `python -m syco prompt` shows each
prompt's fingerprint. Rules and commands for trying changes on the dev items are in
[`prompts/README.md`](prompts/README.md).

## Bruce: OpenAI and DeepSeek

```bash
python -m syco smoke --models api                 # one tiny call each: key, model ID, parameters
python -m syco generate --models api --dry-run    # number of calls and estimated cost
python -m syco generate --models api              # main run; run it again to resume or retry failures
python -m syco status                             # done / failed / cut off, per model and condition
git add outputs/test/generations/gpt-6.2-sol.jsonl outputs/test/generations/deepseek-flash.jsonl
```

- Models (checked 2026-11-18; confirm with `smoke`): `gpt-6.2-sol` with `reasoning_effort: low`,
  and `deepseek-flash` (DeepSeek-V4.1-Flash) with thinking turned off. Alternatives are noted in
  `config.yaml`.
- DeepSeek charges half price off-peak. Peak is 01:00–04:00 and 06:00–10:00 UTC on weekdays,
  which is 5–8 pm and 10 pm–2 am Pacific (Sunday to Thursday evenings).
- 3 datasets × 200 items × 3 conditions = 1,800 calls per model. The dry run estimates about $13
  for gpt-6.2-sol and about $1.60 for DeepSeek at peak prices (assuming 600-token replies).

## Cecilia: local models (Ollama)

1. Install [Ollama](https://ollama.com) and pull the models (several GB each; check the tags at
   ollama.com/library):

   ```bash
   ollama pull qwen3.5:9b
   ollama pull llama3.1:8b
   ```

2. Check that each model answers and that thinking is off (`thinking_chars` must be 0):

   ```bash
   python -m syco smoke --models local
   ```

3. Time a small run, then do the full run (1,800 replies per model). The full run can take hours,
   so keep the laptop awake and plugged in. Running the same command again resumes where it
   stopped.

   ```bash
   python -m syco generate --models local --n 5
   python -m syco generate --models local
   python -m syco status        # the "thinking" column must be 0; "cut off" should be near 0
   ```

4. Commit `outputs/test/generations/qwen3.5-9b.jsonl` and `llama3.1-8b.jsonl`.

The settings are in `config.yaml`: `think: false` (Qwen thinks by default) and `num_ctx: 8192`
(Ollama's default context window is too small for long posts plus the policy prompt). If you run
the models some other way (for example on Colab), add a backend class in `syco/models/` that follows
`ChatModel` in `syco/models/base.py` and register it in `syco/models/__init__.py`. Keep the same
model names, so the output files stay comparable.

### Cecilia: Bolei GPU backend (3B checkpoints)

[`syco/models/bolei_local.py`](syco/models/bolei_local.py) uses the same frozen prompts and
fixed sample IDs as the other backends. The optional [model profile](config.bolei.yaml)
adds Qwen2.5-3B-Instruct and Llama-3.2-3B-Instruct. Each uses BF16, greedy decoding and
the shared 1,024-token limit. The pinned Unsloth Llama distribution has matching Meta
weight shards and supplies its own config and chat template.

Use a compute-visible Python environment with the repository requirements, CUDA PyTorch
2.9 and Transformers 4.57.6. Use an environment without the old Bolei FlashAttention wheel.
Set `HF_HUB_CACHE` to a shared path and stage the pinned snapshots before allocating a GPU.

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
from syco.config import load_config
for model in load_config(model_config="config.bolei.yaml")["models"]:
    if model["provider"] == "bolei":
        snapshot_download(model["model"], revision=model["params"]["revision"],
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
PY
python -m syco fetch-data
python -m syco --model-config config.bolei.yaml generate --models qwen2.5-3b,llama3.2-3b --dry-run
```

Submit from the repository root with the Bolei partition, a wall time, and the shared
Python executable. Append `--n 1 --out outputs/_scratch/bolei` for a nine-answer smoke
test per model. Then submit the same jobs without those arguments for the reportable test.

```bash
sbatch --partition=a5000 --time=12:00:00 scripts/run_bolei.sbatch "$python_bin" qwen2.5-3b
sbatch --partition=a5000 --time=12:00:00 scripts/run_bolei.sbatch "$python_bin" llama3.2-3b
python -m syco --model-config config.bolei.yaml status
```

The test writes `outputs/test/generations/qwen2.5-3b.jsonl` and
`outputs/test/generations/llama3.2-3b.jsonl`, 1,800 answers each. Use
`--model-config config.bolei.yaml` before the `judge` command to select these models.
The standalone full-dataset pilot used the earlier policy text and is a separate run.

## Alice and Yanzi: evaluation

- **Ready:** ELEPHANT's three metrics, with the original prompts copied verbatim
  (`syco/judge/elephant_prompts.py`): `validation`, `indirectness` and `framing`. A label of 1 is
  the sycophantic direction for all three. `config.yaml` (`metrics:`) says which metrics run on
  which dataset.
- **To do:** the policy-matrix rubric. Add a `Rubric` in `syco/judge/rubrics.py` (a commented
  template is at the bottom of that file) and list its name under `metrics:` in `config.yaml`.
  Test it for free with the mock judge first.
- The judge is `claude-haiku-5-5` with thinking off (`judges:` in `config.yaml`).

```bash
python -m syco judge --models mock --judge mock --out outputs/_scratch   # free test
python -m syco judge --models all --dry-run                              # count and estimated cost
python -m syco judge --models all                                        # real run; resumable
python -m syco analyze                                                   # tables for the slides
git add outputs/test/judgments outputs/test/tables
```

- With 3 datasets and 3 conditions there are 4,200 judgments per model (OEQ and AITA-YTA get three
  metrics, SS gets one). `--dry-run` prints the count and estimated cost before you spend anything.
- The judge runs only on responses that exist, so you can start as soon as one model's file is
  pushed.
- Our mid-quarter report says automated scores will be checked against human labels before they
  are reported. Plan a small hand check of each metric.

## Design notes

- **Same items, same settings.** `data/samples.json` fixes the items: 50 dev items per dataset for
  prompt development, and a test list whose first `n_test` items (200) are the reported run. Each
  model gets the same user message in every condition; only the system message differs.
  Temperature is each provider's default. Replies are limited to 1,024 tokens (2,048 for OpenAI,
  because its limit also counts reasoning tokens).
- **Which datasets.** `config.yaml` runs the three sets in our plan doc: `oeq` (general advice,
  emotional support, personal decisions), `aita_yta` (accountability for conduct the community
  judged wrong) and `alp` (the doc's SS, Subjective Statements: assumptions, interpretations and
  uncertain claims; the file is SS.csv). `aita_flip` needs a verdict rubric (YTA/NTA) that isn't
  written yet.
- **Reading the numbers.** For ELEPHANT metrics a lower rate is less sycophantic. A lower
  `validation` rate, though, can also mean less emotional support. `analyze` reports each
  condition's rate and three paired comparisons on the same items (`generic` vs `baseline`,
  `policy` vs `baseline`, `policy` vs `generic`; set under `analysis:` in `config.yaml`), each with
  a 95% bootstrap interval and an exact McNemar p-value.
- **Safety rails.** One output file per model, so teammates never edit the same file. Every record
  carries the prompt's fingerprint, and a test file refuses responses made with a different prompt
  version. Failed calls are saved with their error and retried on the next run. A run stops after
  10 failures in a row.

## Output files

All outputs are JSON Lines: one JSON object per line.

`outputs/<split>/generations/<model>.jsonl`, one line per response:

| field | meaning |
|---|---|
| `dataset`, `item_id` | which item (`item_id` is in `data/samples.json`) |
| `model`, `condition` | model name from `config.yaml`; `baseline`, `generic` or `policy` |
| `system_prompt_sha` | fingerprint of the system prompt sent (`none` if a condition has no system prompt) |
| `response` | the model's answer |
| `error` | `null`, or why the call failed (retried on the next run) |
| `finish_reason`, `input_tokens`, `output_tokens`, `reasoning_tokens`, `served_model`, `latency_s`, `settings`, `created_at` | details for checking and reproducing |

`outputs/<split>/judgments/<judge>/<model>.jsonl`, one line per response and metric:
`dataset`, `item_id`, `model`, `condition`, `metric`, `label`, `judge_reply`, `judge_model`,
`rubric_sha`, `response_sha`, `error`.

`outputs/<split>/tables/<judge>/`:
- `summary.csv`: one row per dataset × metric × model × condition, with `n` and `rate`.
- `comparisons.csv`: one row per dataset × metric × model × comparison, with `condition`,
  `reference`, `n_paired`, `diff` (condition minus reference), `ci_low`, `ci_high` and `mcnemar_p`.
- `summary.md`: both tables, ready to paste.

## Repository layout

```
config.yaml              models, conditions, datasets, judge, metrics (shared by everyone)
prompts/                 the three condition prompts and their rules
data/samples.json        the fixed item IDs (committed); data/raw/ is downloaded, not committed
syco/
  models/                openai_api.py (OpenAI, DeepSeek), ollama_local.py (local), mock.py
  judge/                 rubrics.py, elephant_prompts.py, clients.py (Claude), run.py
  generate.py            runs models x conditions x items
  analyze.py, status.py  tables and progress
tests/                   python -m unittest
outputs/                 committed results; outputs/_scratch/ is for experiments and ignored
```

## Rules

- Never commit `.env` or any key.
- Don't edit the prompts after the test runs start. If one must change, every model is re-run.
- Change `config.yaml` or `data/samples.json` only after the team agrees, and tell everyone.
- Commit only your own output files. Run `git pull` before you start and before you push.

## Troubleshooting

| Message | Fix |
|---|---|
| `Set OPENAI_API_KEY ...` | Add the key to `.env` (copy `.env.example`). |
| `... different system prompt ...` | The prompt changed after test runs. Restore it, or move the old file to `outputs/old/` and re-run that model. |
| `Missing data/raw/...` | Run `python -m syco fetch-data`. |
| `Cannot reach Ollama` | Start the Ollama app (or `ollama serve`). |
| Rate limits or timeouts | Retried automatically. Run the same command again to pick up anything that failed. |
