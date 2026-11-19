# Prompts

`policy_system_prompt.txt` holds the policy system prompt (owner: Jing; two teammates review it).

## Rules

- **The file is the prompt.** Everything in it is sent word for word as the system message, so
  leave out notes, comments and headings you don't want the model to see. Leading and trailing
  blank lines are ignored.
- **One prompt for every model.** All models read this same file. Don't make per-model versions.
- **Develop on the dev split only.** Try drafts on dev items (see the commands below), never on
  test items. The test items are used only for the reported results.
- **Freeze before the main run.** Once the test runs start, the prompt must not change. The
  framework saves a fingerprint of the prompt with every response and refuses to mix versions in
  one test file. If the prompt has to change after test runs started, every model has to be
  re-run.
- Until this file is written, real models refuse to run (the mock model still works).

## Useful commands

```bash
python -m syco prompt                                   # fingerprint and length of each prompt
python -m syco show --dataset aita_yta --split dev --n 3  # read some dev items
python -m syco generate --models deepseek-flash --split dev --n 5   # try the draft on 5 dev items per dataset (needs a DeepSeek key; costs about a cent)
python -m syco compare --model deepseek-flash --split dev --n 5     # baseline vs policy replies, side by side
```

When the prompt is final, post its fingerprint (`python -m syco prompt`) in the group chat, so
everyone can check that their runs used the same one.

## Adding a comparison condition

Put the instruction in a new file here (for example `naive.txt`) and add a line under
`conditions:` in `config.yaml`. Every model then also runs that condition.
