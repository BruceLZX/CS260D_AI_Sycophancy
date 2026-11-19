# Prompts

One file per condition (`conditions:` in `config.yaml`), copied word for word on 11/19 from the
"Final project - System prompt construction" tab of our plan doc. The only change is one missing
space added in the policy prompt ("evidence. Support").

| condition | file | prompt |
|---|---|---|
| `baseline` (A) | `baseline_system_prompt.txt` | neutral baseline |
| `generic` (B) | `generic_system_prompt.txt` | the baseline plus the doc's generic anti-sycophancy instruction, joined with one space |
| `policy` (C) | `policy_system_prompt.txt` | the full policy prompt (owner: Jing; two teammates review it) |

## Rules

- **The file is the prompt.** Everything in it is sent word for word as the system message, so
  leave out notes, comments and headings you don't want the model to see. Leading and trailing
  blank lines are ignored.
- **One prompt for every model.** All models read the same files. Don't make per-model versions.
- **Develop on the dev split only.** Try changes on dev items (see the commands below), never on
  test items. The test items are used only for the reported results.
- **Freeze before the main run.** Once the test runs start, the prompts must not change. The
  framework saves a fingerprint of the prompt with every response and refuses to mix versions in
  one test file. If a prompt has to change after test runs started, every model has to be re-run.

## Useful commands

```bash
python -m syco prompt                                   # fingerprint and length of each prompt
python -m syco show --dataset aita_yta --split dev --n 3  # read some dev items
python -m syco generate --models deepseek-flash --split dev --n 5   # try the prompts on 5 dev items per dataset (needs a DeepSeek key; a few cents)
python -m syco compare --model deepseek-flash --split dev --n 5     # the conditions' replies, side by side
```

When the prompts are final, post their fingerprints (`python -m syco prompt`) in the group chat,
so everyone can check that their runs used the same ones.

## Adding a comparison condition

Put the instruction in a new file here (for example `naive.txt`) and add a line under
`conditions:` in `config.yaml`. Every model then also runs that condition. List the comparisons
you want under `analysis:` in `config.yaml`.
