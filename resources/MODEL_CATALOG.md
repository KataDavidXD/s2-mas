# S²-MAS sharp model family

Model collection: [S²-MAS Models · sharp](https://huggingface.co/collections/ya9919/s-mas-models-finq-and-finl-6ad68009810d66654272d854). Paper: [arXiv:2610.27746](https://arxiv.org/abs/2610.27746), manuscript v52, 21 October 2026.

The sharp family names each model by size and uses suffixes for training method, budget, checkpoint stage or runtime. `sharp-421m` uses the native 421M encoder. `sharp-322m` uses a separate 322M bilingual encoder. `sharp-4b` is the selected 4B LoRA and decision-head model; `sharp-9b` applies that method to the 9B backbone.

Seventeen repository entries package ten budget-matched trained versions, six named checkpoint/runtime entries and one compatibility entry. The paper also compares twelve decision models or systems across six decision-interface routes.

## Six training methods and ten matched versions

| Training method | Upstream architecture | Distinct training examples | Release versions |
|---|---|---|---|
| Encoder SFT | Laya / ModernBERT-large, 421M | 50k and 200k | `sharp-421m-sft-50k`, `sharp-421m-sft-200k` |
| Encoder SFT → RLCD | Same 421M encoder; continue each SFT development-best | A further pass over the same 50k or 200k panel | `sharp-421m-rlcd-50k`, `sharp-421m-rlcd-200k` |
| Encoder SFT → CE | Same 421M encoder; continue the 200k SFT development-best | A further pass over the same 200k panel | `sharp-421m-ce-200k` |
| Frozen 4B backbone + decision head | Qwen3.5-4B-Base | 50k and 200k | `sharp-4b-head-50k`, `sharp-4b-head-200k` |
| 4B LoRA + decision head | Qwen3.5-4B-Base | 50k and 200k | `sharp-4b-lora-50k`, `sharp-4b` |
| 9B LoRA + decision head | Qwen3.5-9B | 200k | `sharp-9b` |

These methods produce 2 + 2 + 1 + 2 + 2 + 1 = 10 selected training versions. The two 200k encoder continuations share the same SFT parent; the additional pass adds training exposures to the existing panel. The original Laya model supplies the reference before additional training.

| Repository | Model and training condition |
|---|---|
| [sharp-421m-sft-50k](https://huggingface.co/ya9919/sharp-421m-sft-50k) | 421M encoder SFT, 50k distinct examples |
| [sharp-421m-sft-200k](https://huggingface.co/ya9919/sharp-421m-sft-200k) | 421M encoder SFT, 200k distinct examples |
| [sharp-421m-rlcd-50k](https://huggingface.co/ya9919/sharp-421m-rlcd-50k) | RLCD continuation of the 50k SFT checkpoint |
| [sharp-421m-rlcd-200k](https://huggingface.co/ya9919/sharp-421m-rlcd-200k) | RLCD continuation of the 200k SFT checkpoint |
| [sharp-421m-ce-200k](https://huggingface.co/ya9919/sharp-421m-ce-200k) | CE continuation of the 200k SFT checkpoint |
| [sharp-4b-head-50k](https://huggingface.co/ya9919/sharp-4b-head-50k) | Frozen 4B backbone and trained decision head, 50k examples |
| [sharp-4b-head-200k](https://huggingface.co/ya9919/sharp-4b-head-200k) | Frozen 4B backbone and trained decision head, 200k examples |
| [sharp-4b-lora-50k](https://huggingface.co/ya9919/sharp-4b-lora-50k) | 4B LoRA and decision head, 50k examples |
| [sharp-4b](https://huggingface.co/ya9919/sharp-4b) | Selected 4B LoRA and decision head, 200k examples |
| [sharp-9b](https://huggingface.co/ya9919/sharp-9b) | 9B LoRA and decision head, 200k examples |

## Full-data checkpoints, runtimes and compatibility

The full-data `sharp-421m` checkpoint completes six SFT + RLCD rounds using 2,925,263 training examples. Its `sharp-421m-mlx` runtime is used in the reported 12M-agent applications. The 322M bilingual checkpoint and its runtime form a separate size branch.

| Repository | Parameters and role |
|---|---|
| [sharp-421m-3rounds](https://huggingface.co/ya9919/sharp-421m-3rounds) | 421M encoder, three-round native checkpoint documentation; no usable weights currently hosted |
| [sharp-421m](https://huggingface.co/ya9919/sharp-421m) | 421M encoder, full-data six-round SFT + RLCD checkpoint |
| [sharp-421m-3rounds-mlx](https://huggingface.co/ya9919/sharp-421m-3rounds-mlx) | Documentation for the three-round MLX format; no usable weights currently hosted |
| [sharp-421m-mlx](https://huggingface.co/ya9919/sharp-421m-mlx) | MLX deployment of the six-round 421M checkpoint |
| [sharp-322m](https://huggingface.co/ya9919/sharp-322m) | 322M bilingual native checkpoint |
| [sharp-322m-mlx](https://huggingface.co/ya9919/sharp-322m-mlx) | MLX deployment of the 322M bilingual checkpoint |
| [sharp-421m-compat](https://huggingface.co/ya9919/sharp-421m-compat) | Compatibility entry for the six-round checkpoint |

Runtime conversions and the compatibility entry package existing weights. The [reported training and test table](../paper/source_data/matched_training_and_test.csv) gives distinct examples, selected exposures and common-test metrics. The [training scripts](../reproduction/training/README.md) implement the ten matched versions; individual model cards retain upstream attribution and component terms.

## Model license

Project model contributions use the [Controlled Model Research License v1.2](models/LICENSE-CONTROLLED-RESEARCH.md): noncommercial scientific use, no redistribution, and no unauthorized reverse engineering, extraction or distillation. Training records are not distributed. Other permissions: yl9919@connect.hku.hk.
