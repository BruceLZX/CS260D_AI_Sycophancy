# Data

## Source

The four ELEPHANT datasets (Cheng et al., "ELEPHANT: Measuring and understanding social
sycophancy in LLMs", ICLR 2026). They are shared through the view-only OSF link in the ELEPHANT
README (github.com/myracheng/elephant). `python -m syco fetch-data` downloads that zip (6.7 MB),
checks its SHA-256, and puts the CSV files in `data/raw/elephant/datasets/`.

**The raw data is not committed.** This repository is public, and the OSF files carry no license,
so we don't redistribute the posts. Everyone downloads them with `fetch-data`. For the same reason,
the output files store item IDs, not the post texts.

| key in config | file | rows | what a model sees |
|---|---|---:|---|
| `aita_yta` | AITA-YTA.csv | 2,000 | an r/AmITheAsshole post the community judged YTA (the poster is at fault) |
| `alp` | SS.csv | 3,777 | an assumption-laden statement (the ELEPHANT paper calls this set ALP; the file is SS.csv) |
| `oeq` | OEQ.csv | 3,027 | an open-ended personal-advice question |
| `aita_flip` | AITA-NTA-FLIP.csv | 1,591 pairs | the original post (poster not at fault) and the same conflict told by the other side |

Rows whose text repeats an earlier row are dropped (17 in OEQ, 11 pairs in AITA-FLIP).

## The fixed sample: `samples.json`

`samples.json` lists the item IDs everyone uses, so every model and every teammate sees exactly
the same items. It was built once with `python -m syco sample` (seed 260) and is committed.

- Each dataset was shuffled once. The first 50 IDs are the **dev** list (prompt development only,
  never reported) and the next 500 are the **test** list.
- A run with `n_test: 200` uses the first 200 IDs of the test list, so a smaller run is always a
  subset of a bigger one.
- For AITA-FLIP an ID is a pair; each pair gives two items (`...-og` and `...-flip`).
- Item IDs are `<dataset>-<row id from the CSV>`, for example `aita_yta-3677` or `alp-1882`.

Don't rebuild `samples.json` unless the team agrees: it would change everyone's items.

To read items: `python -m syco show --dataset aita_yta --split dev --n 3`.
