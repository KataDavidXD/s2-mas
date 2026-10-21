# S²-MAS: A Society-Scale Multi-Agent Simulation Framework with 500 Million Household Persona Agents

[Paper](https://arxiv.org/abs/2610.27746) · [PDF](https://github.com/KataDavidXD/s2-mas/blob/main/paper/S2_MAS_Paper.pdf) · [Project](https://github.com/KataDavidXD/s2-mas) · [Models](https://huggingface.co/collections/ya9919/s-mas-models-sharp) · [Datasets](https://huggingface.co/collections/ya9919/s-mas-dataset-resources)

**12 million household agents. 24 million financial choices. About four hours on 14 GPUs.**

S²-MAS scales persona-driven language-model simulation through three connected components: **500 million synthetic household personas grounded in real-survey data, the sharp decision-model family, and the Genese execution engine**. Household decisions change purchases, cash, debt and payments; those changes become inputs to later decisions.

The completed population run executes **24 million paired-condition choices and 288 million monthly ledger states in 3 h 59 min 17 s**, using eight H100 and six A100 GPUs. A separate 24-month experiment follows consumption and repayment across income and expense shocks.

![S²-MAS: household data, sharp decision models, Genese execution and economic results.](https://raw.githubusercontent.com/KataDavidXD/s2-mas/main/publicity/20261021_sharp/S2_MAS_Poster.png)

## The sharp model family

**sharp names a family; the number identifies its parameter size.** Training and runtime suffixes distinguish variants within a size.

| Model | Parameters | Role |
|---|---:|---|
| [sharp-421m](https://huggingface.co/ya9919/sharp-421m) | 421M | Full-data, six-round encoder decision model trained with SFT + RLCD |
| [sharp-421m-mlx](https://huggingface.co/ya9919/sharp-421m-mlx) | 421M | MLX FP16 runtime used in the 12M-household and monthly applications |
| [sharp-322m](https://huggingface.co/ya9919/sharp-322m) / [sharp-322m-mlx](https://huggingface.co/ya9919/sharp-322m-mlx) | 322M | Bilingual encoder and its MLX runtime; the single-A100 deployment benchmark uses this model |
| [sharp-4b](https://huggingface.co/ya9919/sharp-4b) | 4B | Selected decoder model with LoRA and a decision head, trained on 200k examples |
| [sharp-9b](https://huggingface.co/ya9919/sharp-9b) | 9B | Decoder model with LoRA and a decision head, trained on 200k examples |

The 421M models adapt Laya's ModernBERT-large encoder; the 322M model adapts its multilingual counterpart. The 4B and 9B models use Qwen3.5 backbones. The family shares a direct-choice interface: a household state, a question and complete alternatives go in; an option and its probabilities come out.

The financial post-training comparison places **sharp-421m first in disclosed training scale, with 2,925,263 examples per language**. Its MLX runtime reaches **74.55% accuracy and 38.37% question-macro F1** in consumer financial-choice scenarios, exceeding Jev 1.13 and GLM-5.3-Flash on both measures.

| Consumer financial-choice test | Accuracy (%) | Question-macro F1 (%) |
|---|---:|---:|
| **sharp-421m-mlx** | **74.5544** | **38.3746** |
| GLM-5.3-Flash | 68.8756 | 37.9991 |
| Jev 1.13 | 65.2584 | 37.4793 |

All three rows use the same 240,458 test records and 119 question versions. Question-macro F1 weights question versions equally.

## Six post-training approaches, ten trained variants

The training study compares compact-encoder SFT, further CE training, SFT + RLCD, 4B head-only training, 4B LoRA plus a decision head, and 9B LoRA plus a decision head. The 50k and 200k budgets produce ten trained variants.

| Trained variant | Unique examples | Accuracy (%) | Question-macro F1 (%) |
|---|---:|---:|---:|
| [sharp-421m-sft-50k](https://huggingface.co/ya9919/sharp-421m-sft-50k) | 50,000 | 72.6942 | 34.8511 |
| [sharp-421m-sft-200k](https://huggingface.co/ya9919/sharp-421m-sft-200k) | 200,000 | 73.8753 | 35.9949 |
| [sharp-421m-rlcd-50k](https://huggingface.co/ya9919/sharp-421m-rlcd-50k) | 50,000 | 72.9258 | 34.7841 |
| [sharp-421m-rlcd-200k](https://huggingface.co/ya9919/sharp-421m-rlcd-200k) | 200,000 | 73.9268 | 36.0503 |
| [sharp-421m-ce-200k](https://huggingface.co/ya9919/sharp-421m-ce-200k) | 200,000 | 73.8986 | 36.1732 |
| [sharp-4b-head-50k](https://huggingface.co/ya9919/sharp-4b-head-50k) | 50,000 | 67.3369 | 31.0166 |
| [sharp-4b-head-200k](https://huggingface.co/ya9919/sharp-4b-head-200k) | 200,000 | 67.2841 | 30.8995 |
| [sharp-4b-lora-50k](https://huggingface.co/ya9919/sharp-4b-lora-50k) | 50,000 | 73.7289 | 36.6225 |
| [sharp-4b](https://huggingface.co/ya9919/sharp-4b) | 200,000 | 74.1435 | 36.7343 |
| [sharp-9b](https://huggingface.co/ya9919/sharp-9b) | 200,000 | 74.0919 | 36.6622 |

These variants use BF16 and development-selected checkpoints. The full-data sharp-421m system uses the larger training partition and appears separately above. The results show where adaptation pays: 4B LoRA raises accuracy from head-only training's 67.28% to 74.14%; the compact 200k SFT model reaches 73.88% using 0.733 allocated H100 GPU-hours.

## Genese: decisions, world state and a transaction boundary

Genese gives model inference and world evolution separate responsibilities. **Brain** groups decision requests into bounded partitions and reuses model connections. **World/ECS** stores numeric components in indexed arrays, resolves proposed actions through registered rules and applies the resulting state changes. A model chooses a plan; the world calculates what that plan does to resources and obligations.

Production state, the event outbox and command results commit in **one PostgreSQL transaction**. Failed transitions restore the world-step snapshot. Event delivery proceeds asynchronously after commit. Across workload partitions, **input preparation, inference and ledger calculation overlap**, keeping the stages moving while preserving each household's state history.

Genese combines array-based state, policy/world separation, atomic state–event commits, whole-world rollback, recovery, pooled execution and flow control. The table compares these mechanisms across six engines.

| Native execution capability | Genese | OneSim | OASIS | MiroFish | AgentSociety | MatrAIx |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Array-based ECS storage | **Yes** | N/D | N/D | N/D | N/D | N/D |
| Synchronized state persistence | **Yes** | Yes | No | No | No | No |
| Native action-rule enforcement | **Yes** | Partial | Yes | Yes | Yes | Partial |
| Policy-intent / world-effect separation | **Yes** | No | No | No | No | No |
| Atomic state–event commit | **Yes** | No | No | No | No | No |
| World-step isolation / rollback | **Yes** | No | No | No | Partial | Partial |
| Recovery / resume | **Yes** | No | No | No | Yes | Yes |
| Execution / connection pooling | **Yes** | Yes | No | No | Yes | No |
| Task admission / flow control | **Yes** | Partial | Yes | Yes | Yes | Yes |
| Decision / trial concurrency control | **Yes** | Partial | Yes | Yes | Yes | Yes |

Source: paper Table S13, using the native framework paths examined there. Partial denotes part of the defined mechanism; N/D denotes no documented mechanism. AgentSociety and MatrAIx isolate tasks or trials; the world-step row measures rollback of the complete simulation step. OneSim's synchronized persistence uses its enabled PostgreSQL data module.

![Genese architecture capabilities and controlled engine-speed comparisons.](https://raw.githubusercontent.com/KataDavidXD/s2-mas/main/publicity/20261021_sharp/S2_MAS_Engine_Comparison.png)

## Engine speed and model throughput

With **1,000 agents and the same sharp-421m allocation on one T4**, Genese completes a round of accepted decisions and writeback in **6.17 seconds**, compared with **7.45 seconds for OASIS** and **7.70 seconds for OneSim**. Elapsed time falls by 17% and 20%, respectively.

The immediate-response backend measures execution overhead for the same 1,000-agent choice-and-writeback workload:

| Framework / tested path | Median round time (s) |
|---|---:|
| **Genese, parallel partitions** | **2.743** |
| AgentSociety | 5.818 |
| OneSim | 6.223 |
| OASIS | 7.386 |
| MiroFish | 11.255 |
| MatrAIx, parallel questionnaire adapter | 1,062.153 |

The table uses twenty randomized interleaved blocks per configuration; initialization is measured separately. Genese completes these rounds 2.12–4.10 times as fast as AgentSociety, OneSim, OASIS and MiroFish.

For single-request model deployment, **sharp-322m-mlx processes 36,955 shared input tokens/s at 9.10 ms mean latency**. Its input throughput is **22.72× Jev API** and **2.42× the same checkpoint in PyTorch**.

| Single-request deployment | Mean latency (ms) | Shared input tokens/s |
|---|---:|---:|
| **sharp-322m-mlx** | **9.10** | **36,955** |
| sharp-322m, PyTorch | 22.06 | 15,247 |
| Jev 1.13 API | 206.77 | 1,627 |

This timing uses three Chinese financial inputs and sixty timed calls per arm, with batch size and concurrency one. Local models run in FP16 on one A100 PCIe 40GB; Jev time includes the API round trip.

## Consumption, cash and repayment over time

Eight financial task families share the household-state, question and options interface: spending adjustment, medical funding, purchase funding, loan terms, liquid saving and risky assets, debt-payment allocation, repayment difficulty and liquidity shortfalls.

The 12M-household purchase experiment raises both the purchase goal and approved credit limit from CNY 20,000 to CNY 30,000 for the same households. Twelve-month consumption rises by **CNY 2,887 per household**, closing cash falls by **CNY 929**, and closing debt rises by **CNY 2,072**. Age and income distinguish consumption responses; education, employment and household-size differences persist after age–income standardization. Much of the variation comes from households moving from deferral to purchase.

A separate **10,000-household panel runs for 24 months**, crossing two purchase goals with five shock paths and an unshocked reference. The same credit product serves both goals. Persistent income loss draws down reserves and narrows the consumption gap between goals; sustained gains widen it. A one-time necessary expense raises paid necessary expenditure while reducing optional purchases. Realized payments and arrears change subsequent bank offers, linking current choices to future opportunities.

## Research materials

The model collection contains ten matched trained variants, full-data checkpoints, runtime formats and a compatibility entry. The two three-round entries provide documentation. The independent synthetic companion supplies **2M household states and 32,000 unlabelled scenario inputs**; a 320-input subset provides a smaller starting point.

[sharp models](https://huggingface.co/collections/ya9919/s-mas-models-sharp) · [Synthetic companion](https://huggingface.co/datasets/ya9919/s2-mas-synthetic-household-companion) · [Example inputs](https://huggingface.co/datasets/ya9919/s2-mas-synthetic-finance-examples) · [Wikidata context](https://huggingface.co/datasets/ya9919/s2-mas-wikidata-context) · [Posters](https://github.com/KataDavidXD/s2-mas/tree/main/publicity/20261021_sharp)

Public code covers scientific plotting, decision-model training and evaluation, and independent synthetic companion generation. **The Genese engine implementation remains private.** Models and companion data use their component research terms; Wikidata facts are CC0, authored task designs CC BY 4.0 and public Python code Apache-2.0.

Research collaboration: **yli9919@hku.hk**.
