# Model releases and interfaces

The comparison uses public releases available on 2026-10-14 and the explicitly requested Jev 1.13 hosted model. Project Fin-Laya artifacts are currently private. Upstream resources follow their own licenses and access conditions.

| Benchmark name | Official release | Backbone / interface | Evaluated release |
|---|---|---|---|
| Base Laya | [Laya System One](https://github.com/LeonaDavinci/laya-system-one) | ModernBERT-large with a typed-choice decision head | Original base, published MLX conversion |
| Fin-Laya 6-round | [ya9919/fin-laya-mlx-6-rounds](https://huggingface.co/ya9919/fin-laya-mlx-6-rounds) | Financially adapted Laya decision head and encoder | rlcd-2, MLX |
| OpenDecider Nano | [manjunathshiva/opendecider-nano](https://huggingface.co/manjunathshiva/opendecider-nano), [official interface](https://github.com/manjunathshiva/opendecider) | Ettin encoder and independent option scoring | Public main, OpenDecider 0.8.1 |
| Von | [wfzyx/von](https://huggingface.co/wfzyx/von) | ModernBERT-based option decision model | von-1.3.0, von-sdk 1.3.7 |
| Liquid d1 | [LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) | Released multimodal structured-decision model, text interface | Public main |
| GLiNER2.5-Decide | [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) | Dynamic classification schema and candidate-label scoring | Public main, gliner2 2.1.0 |
| Kev-4B | [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b), [official code](https://github.com/jaredpalmer/kev) | Qwen3.5-4B with structured option probabilities | Kev-4B v1.0, official Torch row batching |
| Clef-Flash | [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) | Qwen3.5-9B with a joint schema head | Public main |
| Tev1-4B | [togethercomputer/Tev1-4B-experimental](https://huggingface.co/togethercomputer/Tev1-4B-experimental), [official recipe](https://github.com/togethercomputer/tev1) | Qwen3.5-4B decision SFT, standard next-token head and option-letter interface | Public main, experimental release |
| Laya-Finance | [hravi/laya-finance](https://huggingface.co/hravi/laya-finance) | ModernBERT-large Laya fine-tuned for financial text decisions | Third public checkpoint |
| Jev | [TypeSafe SystemOne](https://typesafe.ai/) | Hosted typed-choice decision API | Explicit `jev-1.13.0` request |

The paper's Tables S11-S12 list related decision interfaces and financial adaptation resources. Backend and arithmetic labels in the result table describe the evaluated runtime separately from model architecture.
