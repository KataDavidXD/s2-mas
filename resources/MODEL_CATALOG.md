# S²-MAS model repositories

Current collection: [S²-MAS models](https://huggingface.co/collections/ya9919/s-mas-models-v43).

The paper compares six decision-model routes and twelve models or systems. Seventeen repository entries package ten trained release versions, six named checkpoint/runtime entries and one compatibility entry. Two three-round entries currently provide documentation only and contain no usable model weights. Runtime conversions and aliases do not add independent matched training runs.

## Ten trained release versions

| Repository | Role |
|---|---|
| [laya-finance-matched-sft-50k](https://huggingface.co/ya9919/laya-finance-matched-sft-50k) | Native Laya SFT, 50k distinct examples |
| [laya-finance-matched-sft-200k](https://huggingface.co/ya9919/laya-finance-matched-sft-200k) | Native Laya SFT, 200k distinct examples |
| [laya-finance-matched-sft-rlcd-50k](https://huggingface.co/ya9919/laya-finance-matched-sft-rlcd-50k) | Additional RLCD pass after matching 50k SFT |
| [laya-finance-matched-sft-rlcd-200k](https://huggingface.co/ya9919/laya-finance-matched-sft-rlcd-200k) | Additional RLCD pass after matching 200k SFT |
| [laya-finance-matched-sft-continued-200k](https://huggingface.co/ya9919/laya-finance-matched-sft-continued-200k) | CE continuation after 200k SFT |
| [qwen3.5-4b-base-finance-head-50k](https://huggingface.co/ya9919/qwen3.5-4b-base-finance-head-50k) | Qwen4B Base, head-only, 50k |
| [qwen3.5-4b-base-finance-head-200k](https://huggingface.co/ya9919/qwen3.5-4b-base-finance-head-200k) | Qwen4B Base, head-only, 200k |
| [qwen3.5-4b-base-finance-lora-head-50k](https://huggingface.co/ya9919/qwen3.5-4b-base-finance-lora-head-50k) | Qwen4B Base, LoRA and decision head, 50k |
| [qwen3.5-4b-base-finance-lora-head-200k](https://huggingface.co/ya9919/qwen3.5-4b-base-finance-lora-head-200k) | Qwen4B Base, LoRA and decision head, 200k |
| [qwen3.5-9b-finance-lora-head-200k](https://huggingface.co/ya9919/qwen3.5-9b-finance-lora-head-200k) | Qwen9B, LoRA and decision head, 200k |

## Full-data checkpoints, formats and compatibility

| Repository | Role |
|---|---|
| [fin-laya-3-rounds](https://huggingface.co/ya9919/fin-laya-3-rounds) | Documentation-only entry for the native three-round checkpoint; no usable weights are currently hosted |
| [fin-laya-6-rounds](https://huggingface.co/ya9919/fin-laya-6-rounds) | native continuation |
| [fin-laya-mlx-3-rounds](https://huggingface.co/ya9919/fin-laya-mlx-3-rounds) | Documentation-only entry for the paired three-round MLX format; no usable weights are currently hosted |
| [fin-laya-mlx-6-rounds](https://huggingface.co/ya9919/fin-laya-mlx-6-rounds) | MLX deployment format and full-data system reference |
| [fin-laya-multilingual](https://huggingface.co/ya9919/fin-laya-multilingual) | Multilingual native checkpoint |
| [fin-laya-multilingual-mlx](https://huggingface.co/ya9919/fin-laya-multilingual-mlx) | Multilingual MLX deployment format |
| [fin-laya](https://huggingface.co/ya9919/fin-laya) | Compatibility entry |

The [reported training and test table](../paper/source_data/matched_training_and_test.csv) gives distinct examples, selected exposures and common-test metrics. The [paper](../paper/S2_MAS_Paper.pdf) reports full-data systems separately. Individual repositories carry their component terms and upstream model notices.

## Model license

Project model contributions in the listed repositories use the [Controlled Model Research License v1.2](models/LICENSE-CONTROLLED-RESEARCH.md): noncommercial scientific use, no redistribution, and no unauthorized reverse engineering, extraction or distillation. Model parameters and reported results are unchanged. Training records are not distributed. Other permissions: yli9919@hku.hk.
