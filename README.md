# S²-MAS: A Society-Scale Multi-Agent Simulation Framework with 500 Million Household Persona Agents

**500 million household personas, 36 fields, six decision-model routes, ten trained release versions and twelve benchmarked models or systems.** S²-MAS connects reusable household decisions with monthly financial state evolution.

sharp-421m uses the **largest disclosed SFT + RLCD training set among the financial decision models compared in the paper: 2,925,263 training examples**. S²-MAS completes paired-condition simulation of **12,000,000 distinct synthetic household agents—the largest model-inferred MAS population among the completed studies compared in the paper**.

- **74.5544% accuracy · 38.3746% macro-F1**, on 240,458 shared test records spanning 119 question versions and 8,275 groups.
- **9.10 ms · 22.72× faster than the Jev API · 2.42× faster than the same checkpoint in PyTorch**, measured for sharp-322m-mlx on three fixed Chinese inputs, 60 timed calls per arm, batch and concurrency 1, with local models on one A100 PCIe 40GB. Timing includes preparation and synchronized prediction or API round trip.

**Completed scale run: 12M distinct agents · 24M paired-condition choices · 288M ledger states · 3 h 59 m 17 s · 14 GPUs (8 H100 80GB + 6 A100 40GB).**

[Latest manuscript](paper/S2_MAS_Paper.pdf) · [LaTeX and figures](paper/) · [Reported aggregates](paper/source_data/) · [中文](README_ZH.md)

Manuscript v54 · 22 October 2026 · [arXiv:2610.27746](https://arxiv.org/abs/2610.27746).

| Resource | Entry point and scope |
|---|---|
| Paper and scientific scripts | [Paper resources and aggregate plotting](paper/README.md) and [training and evaluation scripts](reproduction/training/README.md) |
| Decision models | [17-repository model catalog](resources/MODEL_CATALOG.md) · [sharp model collection](https://huggingface.co/collections/ya9919/s-mas-models-sharp): ten trained release versions, trained checkpoints, runtime formats and one compatibility entry |
| Synthetic companion | [Generation code, schema and terms](resources/public_companion/README.md); 2M independently constructed financial personas and 32k unlabeled inputs. The complete data are in the [Hugging Face dataset](https://huggingface.co/datasets/ya9919/s2-mas-synthetic-household-companion) · [dataset resources collection](https://huggingface.co/collections/ya9919/s-mas-dataset-resources); paper and code bundles are in [GitHub releases](https://github.com/KataDavidXD/s2-mas/releases) |
| Wikidata component | [Actual CC0 knowledge files](resources/wikidata/README.md): 38 entities and 66 parent/type relations for assigned contexts |


The framework, decision-model routes and conditional shock applications are illustrated below.

![S2-MAS framework](paper/figures/F1.png)

![Decision-model training routes](paper/figures/DECISION_TRAINING_ROUTES.png)

![Conditional shock applications](paper/figures/SHOCK_OVERVIEW.png)

The synthetic companion and Wiki context are separate from the paper's 12M execution cohort and its five-shock monthly extension and from existing model training. The 500M headline describes the framework's persona resource.

Public scientific plotting, training and evaluation scripts use [Apache-2.0](LICENSE). **GENESE engine implementation remains private.** Companion records use [research-use data terms](resources/public_companion/LICENSE-DATA.md); Wikidata facts remain [CC0](resources/wikidata/LICENSE-CC0.md), and the existing authored task designs retain CC BY 4.0. Project model contributions use [controlled research terms](resources/models/LICENSE-CONTROLLED-RESEARCH.md): noncommercial use, no redistribution and no unauthorized reverse engineering.

Research collaboration: [yli9919@hku.hk](mailto:yli9919@hku.hk).

[Blog](https://huggingface.co/blog/ya9919/s2-mas) · [Posters and engine comparison](publicity/20261021_sharp/)
