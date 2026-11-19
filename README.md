# Policy prompt vs. sycophancy — experiment code (CS 260E)

We test whether our policy framework, given to a model as a system prompt, changes how the model
answers personal-advice requests. Every model answers the same items twice, with no system prompt
(`baseline`) and with the policy prompt (`policy`). A Claude judge then scores the answers with
ELEPHANT's metrics (and our own rubric), and we compare the two conditions on the same items.

```
data/samples.json (fixed items) ─┐
                                 ├─► generate ─► outputs/test/generations/<model>.jsonl
prompts/policy_system_prompt.txt ┘                      │
                                              judge ◄───┘ ─► outputs/test/judgments/claude/<model>.jsonl
                                                                  │
                                              analyze ◄───────────┘ ─► outputs/test/tables/claude/summary.md
```

## Who does what (Nov 18 meeting)

| Part | Owner | Where in this repo | Due |
|---|---|---|---|
| Policy system prompt (2 others review) | Jing | `prompts/policy_system_prompt.txt` | Thu 11/19 |
| API models: OpenAI, DeepSeek | Bruce | `syco/models/openai_api.py` → `outputs/test/generations/` | Sat 11/21 |
| Local models: Qwen, Llama (Ollama) | Cecilia | `syco/models/ollama_local.py` → `outputs/test/generations/` | Sat 11/21 |
| Evaluation: Claude judge, ELEPHANT metrics, policy-matrix rubric | Alice, Yanzi | `syco/judge/` → `outputs/test/judgments/`, `outputs/test/tables/` | Sun 11/22 |
| Slides | everyone | | draft Mon 11/23, final Tue 11/24 |

What already works: data download and the fixed sample, the OpenAI and DeepSeek runner, the
Ollama runner (tested only against a fake server so far), the Claude judge with ELEPHANT's three
metrics, the summary tables, and a mock model and mock judge that let you try every step for free.
What still needs you: the policy prompt, real runs, the policy-matrix rubric, and the choice of
datasets (see [Design notes](#design-notes)).

## Setup (everyone, about 5 minutes)

Python 3.10 or newer.

```bash
git clone https://github.com/BruceLZX/CS260D_AI_Sycophancy.git
cd CS260D_AI_Sycophancy
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m syco fetch-data            # ELEPHANT datasets (6.7 MB) into data/raw/
python -m unittest                   # about 25 tests, a few seconds, no keys needed
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

## Jing: the policy prompt

Write the prompt into `prompts/policy_system_prompt.txt`, replacing everything in it. The whole
file is sent word for word as the system message to every model. Until it is written, real models
refuse to run. Rules and commands for trying drafts on the dev items are in
[`prompts/README.md`](prompts/README.md). Short version:

```bash
python -m syco show --dataset aita_yta --split dev --n 3                 # read dev items
python -m syco generate --models deepseek-flash --split dev --n 5        # try a draft (DeepSeek key; about a cent)
python -m syco compare --model deepseek-flash --split dev --n 5          # baseline vs policy, side by side
python -m syco prompt                                                    # the prompt's fingerprint
```

When it is final, post the fingerprint from `python -m syco prompt` in the group chat. After the
test runs start, the prompt must not change.

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
- Estimated cost for 2 datasets × 200 items × 2 conditions: about $6 for gpt-6.2-sol and under $1
  for DeepSeek.

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

3. Time a small run, then do the full run. The full run can take hours, so keep the laptop awake
   and plugged in. Running the same command again resumes where it stopped.

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

- Judging is cheap: about $1–2 for all four models on two datasets.
- The judge runs only on responses that exist, so you can start as soon as one model's file is
  pushed.
- Our mid-quarter report says automated scores will be checked against human labels before they
  are reported. Plan a small hand check of each metric.

## Design notes

- **Same items, same settings.** `data/samples.json` fixes the items: 50 dev items per dataset for
  prompt development, and a test list whose first `n_test` items (200) are the reported run. Each
  model gets the same user message in both conditions; only the system message differs.
  Temperature is each provider's default. Replies are limited to 1,024 tokens (2,048 for OpenAI,
  because its limit also counts reasoning tokens).
- **Which datasets.** `config.yaml` runs `aita_yta` and `alp`, the two our data section calls most
  relevant. Both test harmful agreement. Our research question also asks whether emotional
  support is kept, and only `oeq` (with the `validation` metric) tests that. Swapping one dataset,
  or adding `oeq`, is a one-line change in `config.yaml`; decide before the main runs start.
  `aita_flip` needs a verdict rubric (YTA/NTA) that isn't written yet.
- **Reading the numbers.** For ELEPHANT metrics a lower rate is less sycophantic. A lower
  `validation` rate, though, can also mean less emotional support. `analyze` reports each rate, the
  change from baseline on the same items with a 95% bootstrap interval, and an exact McNemar p-value.
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
| `model`, `condition` | model name from `config.yaml`; `baseline` or `policy` |
| `system_prompt_sha` | fingerprint of the system prompt sent (`none` for baseline) |
| `response` | the model's answer |
| `error` | `null`, or why the call failed (retried on the next run) |
| `finish_reason`, `input_tokens`, `output_tokens`, `reasoning_tokens`, `served_model`, `latency_s`, `settings`, `created_at` | details for checking and reproducing |

`outputs/<split>/judgments/<judge>/<model>.jsonl`, one line per response and metric:
`dataset`, `item_id`, `model`, `condition`, `metric`, `label`, `judge_reply`, `judge_model`,
`rubric_sha`, `response_sha`, `error`.

`outputs/<split>/tables/<judge>/summary.csv` and `summary.md`: one row per dataset × metric × model
× condition, with `n`, `rate`, `diff_vs_baseline`, `ci_low`, `ci_high` and `mcnemar_p`.

## Repository layout

```
config.yaml              models, conditions, datasets, judge, metrics (shared by everyone)
prompts/                 the policy system prompt (Jing) and its rules
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
- Don't edit the policy prompt after the test runs start. If it must change, every model is re-run.
- Change `config.yaml` or `data/samples.json` only after the team agrees, and tell everyone.
- Commit only your own output files. Run `git pull` before you start and before you push.

## Troubleshooting

| Message | Fix |
|---|---|
| `Set OPENAI_API_KEY ...` | Add the key to `.env` (copy `.env.example`). |
| `... still holds the placeholder text` | The policy prompt isn't written yet; only the mock model runs. |
| `... different system prompt ...` | The prompt changed after test runs. Restore it, or move the old file to `outputs/old/` and re-run that model. |
| `Missing data/raw/...` | Run `python -m syco fetch-data`. |
| `Cannot reach Ollama` | Start the Ollama app (or `ollama serve`). |
| Rate limits or timeouts | Retried automatically. Run the same command again to pick up anything that failed. |
