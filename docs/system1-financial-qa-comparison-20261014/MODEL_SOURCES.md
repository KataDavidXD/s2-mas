# Model releases and interfaces

Paper: [arXiv:2610.27746](https://arxiv.org/abs/2610.27746) · manuscript v58 · 22 October 2026.

The comparison combines the public decision-model releases evaluated from 14 October 2026 with the hosted API comparisons reported through 22 October 2026. The [sharp model catalog](../../resources/MODEL_CATALOG.md) lists project releases and their controlled research terms. Upstream resources follow their own licenses and access conditions.

| Benchmark name | Official release | Backbone / interface | Evaluated release |
|---|---|---|---|
| Base Laya | [Laya System One](https://github.com/LeonaDavinci/laya-system-one) | ModernBERT-large with a typed-choice decision head | Original base, published MLX conversion |
| sharp-421m-mlx | [sharp-421m-mlx](https://huggingface.co/ya9919/sharp-421m-mlx) | Financially adapted Laya decision head and encoder | rlcd-2, MLX |
| OpenDecider Nano | [manjunathshiva/opendecider-nano](https://huggingface.co/manjunathshiva/opendecider-nano), [official interface](https://github.com/manjunathshiva/opendecider) | Ettin encoder and independent option scoring | Public main, OpenDecider 0.8.1 |
| Von | [wfzyx/von](https://huggingface.co/wfzyx/von) | ModernBERT-based option decision model | von-1.3.0, von-sdk 1.3.7 |
| Liquid d1 | [LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) | Released multimodal structured-decision model, text interface | Public main |
| GLiNER2.5-Decide | [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) | Dynamic classification schema and candidate-label scoring | Public main, gliner2 2.1.0 |
| Kev-4B | [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b), [official code](https://github.com/jaredpalmer/kev) | Qwen3.5-4B with structured option probabilities | Kev-4B v1.0, official Torch row batching |
| Clef-Flash | [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) | Qwen3.5-9B with a joint schema head | Public main |
| Tev1-4B | [togethercomputer/Tev1-4B-experimental](https://huggingface.co/togethercomputer/Tev1-4B-experimental), [official recipe](https://github.com/togethercomputer/tev1) | Qwen3.5-4B decision SFT, standard next-token head and option-letter interface | Public main, experimental release |
| Laya-Finance | [hravi/laya-finance](https://huggingface.co/hravi/laya-finance) | ModernBERT-large Laya fine-tuned for financial text decisions | Third public checkpoint |
| Jev | [TypeSafe SystemOne](https://typesafe.ai/) | Hosted typed-choice decision API | Explicit `jev-1.13.0` request |

The hosted API rows include GLM-5.3-Flash, GPT-6-Luna and DeepSeek-V4.1-Flash on the full 240,458-record test, and GPT-6-Astra on the fixed 14,441-record sample. Their backend and scoring scope are reported in Table 4 and Supplementary Methods M5.

The paper's Tables S7-S8 list related decision interfaces and financial adaptation resources. Backend and arithmetic labels in the result table describe the evaluated runtime separately from model architecture.

Parameter entries were checked against official documentation and public checkpoint metadata on 22 October 2026. [Model sizes and their reporting basis](../../paper/source_data/SYSTEM1_model_parameters.csv) distinguish complete checkpoint totals, nominal backbone classes and MoE components. GLM-5.3-Flash reports 320B total / 18B active. DeepSeek-V4.1-Flash reports a 552B backbone plus 196B Engram parameters, with 8B active during prefill and 16B during decode. Laya-Finance contains 421,293,830 parameters. GLiNER2.5-Decide contains 486,444,053 parameters in its complete checkpoint; its card uses the 340M nominal label. The official Jev and GPT-6 model documentation checked here does not provide parameter counts.
