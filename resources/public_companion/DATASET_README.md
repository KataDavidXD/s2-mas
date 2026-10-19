---
pretty_name: S²-MAS Synthetic Household Companion
license: other
license_name: s2-mas-new-synthetic-research-v1
license_link: LICENSE-DATA.md
language:
  - en
configs:
  - config_name: synthetic_personas
    default: true
    data_files:
      - split: synthetic
        path: data/personas/*.jsonl.gz
  - config_name: clean_scenario_inputs
    data_files:
      - split: example
        path: data/scenario_inputs.jsonl.gz
tags:
  - synthetic
  - financial-decisions
  - household-personas
---

# S²-MAS Synthetic Household Companion

A separate research companion for the **S²-MAS framework with its reported 500M persona resource**. This version contains **2,000,000 independently constructed synthetic financial households** in **40 gzip JSONL shards (50,000 records each)**, plus **32,000 unlabeled inputs** from the first 2,000 households × eight authored tasks × two conditions. The first-20-household/320-input example is a subset. No model answers or monthly ledgers are included.

Persona shards total **166,263,000 compressed bytes** (2,048,368,765 uncompressed); the 32,000 inputs total **11,300,790 compressed bytes**. Exact per-file counts are in `generation_info.json`.

An independent RNG assigns new household financial states using real-survey aggregate expectation scales. Wikidata provides 38 entities and 66 relations for assigned geography labels and parent/type context. See [NOTICE.md](NOTICE.md) for scope and [SCHEMA.md](SCHEMA.md) for units.

The persona/scenario records use [research-use data terms](LICENSE-DATA.md). The Wikidata files remain **CC0**, the existing authored design file **CC BY 4.0**, and Python code **Apache-2.0**. This companion is separate from the paper's execution populations and model training. Research collaboration: **yli9919@hku.hk**.

To construct another independent version with Python's standard library, run `python code/build_release.py --count 20 --seed 2026101919 --output regenerated`; the output directory must not already contain generated data. Validate saved output with `python code/validate_release.py --data-dir regenerated`.


[Paper and project](https://github.com/KataDavidXD/s2-mas).
