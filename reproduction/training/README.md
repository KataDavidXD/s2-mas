# Decision-model training and common-test evaluation

This directory implements the ten budget-matched sharp training versions reported
in [arXiv:2610.27746](https://arxiv.org/abs/2610.27746), manuscript v52,
21 October 2026. They cover six methods: 421M encoder SFT at 50k/200k unique
rows, SFT → RLCD at both budgets, SFT200k → CE at 200k, frozen 4B backbone
plus head at both budgets, 4B LoRA plus head at both budgets, and 9B LoRA plus
head at 200k. The [model catalog](../../resources/MODEL_CATALOG.md) maps these
methods to `sharp-421m-*`, `sharp-4b-*`, `sharp-4b` and `sharp-9b` releases.
The original Laya checkpoint supplies the reference before additional training.

`sharp-421m` names the 421M encoder branch. The full-data six-round checkpoint
and its MLX runtime are listed separately from these matched-budget versions;
the bilingual 322M checkpoint is `sharp-322m`.

For an exported sharp-4b or sharp-9b package, inference does not require training or
optimizer files:

```python
from qwen_decision import DecisionPredictor

predictor = DecisionPredictor.from_pretrained(
    "models/decision-package", "Qwen/Qwen3.5-4B-Base", device="cuda")
row = {
    "state": {"monthly_income": 5000, "cash_reserve": 10000},
    "qid": "example_spending",
    "question": {"instructions": "Choose a discretionary spending plan.",
                 "criteria": {"A": "Maintain planned spending", "B": "Reduce spending"}},
    "candidate_ids": ["A", "B"],
}
result = predictor.predict(row)
print(result["prediction"], result["probabilities"])
```

This is an original synthetic API example. For a 9B decision package,
supply its named `Qwen/Qwen3.5-9B` upstream backbone.

Use Python 3.12 and install `requirements.txt` with the appropriate PyTorch CUDA
build. The native renderer, model, CE and RLCD objective come from
[Laya 0.4.2](https://pypi.org/project/laya/0.4.2/). Qwen training uses the
Transformers Qwen3.5 text backbone and PEFT. Optional Flash Linear Attention and
causal-conv1d kernels affect throughput; the training scripts do not download
custom kernels. Training requires CUDA; data preparation, prediction merging,
method selection and paired statistics run on CPU.

## Data and models

The input interface accepts labeled decision records supplied under the access
and use permissions for those records. Training and evaluation records are not
included in this directory. `manifest.example.json` illustrates the paper's fixed
partition sizes using generic gzip JSONL paths; replace the paths and counts with
your authorized input files.

Each native record contains `state`, one versioned question and its complete
ordered `criteria`, the expected choice, and metadata `row_id`, `group` and
`task_family`. The caller supplies ordinary row, household/group and question IDs.
`task_family` is a user-defined task label, such as `payment_planning`; it is not
part of the rendered model input. Preparation copies `row_id` without decoding it
or looking up an external record. Row IDs support matching; group IDs support
split separation and paired statistics.

From this directory, prepare common Qwen rows and map the same rows to native
Laya. Both forms retain the complete state, question, alternatives and target.

```sh
python prepare_data.py --manifest data/manifest.json --output prepared/qwen --prepare-test
python prepare_native.py --source-dir prepared/qwen --output prepared/laya --prepare-test
```

The paper's fixed samples use proportional task-family/question strata with seed
20261018; 50k is nested in 200k. The checkpoint-selection set is dev20k. Full dev
has 412,873 rows and 829 question versions; the common final test has 240,458
rows and 119 versions. A new input population does not reproduce those results.
Test records never enter training, budget or method selection. Preparation checks
ordinary IDs, label legality and group separation. Native input windows are
measured from complete inputs; an insufficient window or an option exceeding
the native 48-token cap is rejected.

Model paths are local Hugging Face snapshots. The starting models are
`Qwen/Qwen3.5-4B-Base`, `Qwen/Qwen3.5-9B` and `convaiinnovations/laya`. Project
model cards identify the ten dev-best sharp releases and their controlled
research-use terms. Existing upstream licenses remain effective. Qwen head/adapter packages need
their named upstream backbone; native Laya packages include encoder, tokenizer
and decision weights. Upstream checkpoint histories differ: the model comparison
controls additional downstream rows, rather than claiming identical pretraining.

## Train one reported version

`recipes.json` gives the actual parameters. `run_recipe.py` launches one selected
foreground training job and has no server or background queue. The paper uses
eight ranks: Qwen LoRA global batch32; native Laya global batch768. Changing rank
count changes the update geometry unless the batch/accumulation is adjusted.

```sh
python run_recipe.py --list
python run_recipe.py --recipe sharp-4b-head-50k --data prepared/qwen --model models/qwen4b-base --output runs/sharp-4b-head-50k
python run_recipe.py --recipe sharp-4b-lora-50k --data prepared/qwen --model models/qwen4b-base --output runs/sharp-4b-lora-50k
python run_recipe.py --recipe sharp-4b-head-200k --data prepared/qwen --model models/qwen4b-base --output runs/sharp-4b-head-200k
python run_recipe.py --recipe sharp-4b --data prepared/qwen --model models/qwen4b-base --output runs/sharp-4b
```

Qwen independently scores each candidate using the complete respondent state,
question, all options and the candidate in focus. It takes the last non-padding
text hidden state, normalizes it with L2, applies the shared linear decision head,
and normalizes over the legal candidate set. Head-only freezes the backbone and
caches candidate features; LoRA jointly trains rank16 attention adapters and the
head with observed-choice CE. The two methods retain the same input and head.
Head-only permits multiple cached-head epochs with dev patience3; LoRA permits
one pass. Unique rows and actual repeated exposures are reported separately.

Evaluate the two 200k Qwen best checkpoints on **full dev** using the direct CLI:

```sh
torchrun --standalone --nproc_per_node=8 train_qwen.py --model-path models/qwen4b-base --model-name Qwen3.5-4B-Base --method head_only --checkpoint runs/sharp-4b-head-200k/best --evaluate-path prepared/qwen/dev.full.jsonl --output evaluations/head_full_dev --batch-size 512 --feature-batch-size 128
torchrun --standalone --nproc_per_node=8 train_qwen.py --model-path models/qwen4b-base --model-name Qwen3.5-4B-Base --method lora --checkpoint runs/sharp-4b/best --evaluate-path prepared/qwen/dev.full.jsonl --output evaluations/lora_full_dev --batch-size 2
python select_method.py --head-evaluation evaluations/head_full_dev/evaluation.json --lora-evaluation evaluations/lora_full_dev/evaluation.json --output method_selection.json
```

The method is selected by full-dev mean question macro-F1, then accuracy. The
reported selected method is LoRA; run `sharp-9b` with `models/qwen9b`.
Each model's saved best is selected on dev20k, with no test-based model choice.

Train native SFT first, then continue the named SFT best:

```sh
python run_recipe.py --recipe sharp-421m-sft-50k --data prepared/laya --model models/laya-original --output runs/sharp-421m-sft-50k
python run_recipe.py --recipe sharp-421m-sft-200k --data prepared/laya --model models/laya-original --output runs/sharp-421m-sft-200k
python run_recipe.py --recipe sharp-421m-rlcd-50k --data prepared/laya --model models/laya-original --parent runs/sharp-421m-sft-50k/best --output runs/sharp-421m-rlcd-50k
python run_recipe.py --recipe sharp-421m-rlcd-200k --data prepared/laya --model models/laya-original --parent runs/sharp-421m-sft-200k/best --output runs/sharp-421m-rlcd-200k
python run_recipe.py --recipe sharp-421m-ce-200k --data prepared/laya --model models/laya-original --parent runs/sharp-421m-sft-200k/best --output runs/sharp-421m-ce-200k
```

The two 200k continuations start from the same SFT200k best, reset optimizer and
scheduler, and use the same data order, learning rates and one-pass budget. RLCD
uses four noisy-logit samples, proper-score rewards and a policy-gradient term
plus soft CE; CE continuation uses soft CE alone. All stages select best by dev
macro-F1 then accuracy. Stage/parent/cumulative exposures and unique rows remain
distinct: another pass over the same 200k panel adds exposures, not new unique
rows. Resume only a run's own `last` using the direct training CLI's `--resume`.

## Final common test and paired accuracy/F1

After development selection, evaluate every saved version on the same prepared
shared test. Examples for Qwen and native Laya:

```sh
torchrun --standalone --nproc_per_node=8 train_qwen.py --model-path models/qwen4b-base --model-name Qwen3.5-4B-Base --method lora --checkpoint runs/sharp-4b/best --evaluate-path prepared/qwen/test.shared119.jsonl --output evaluations/sharp-4b --batch-size 2
torchrun --standalone --nproc_per_node=8 train_laya.py --model-path models/laya-original --checkpoint runs/sharp-421m-rlcd-200k/best --evaluate-path prepared/laya/test.shared119.jsonl --output evaluations/sharp-421m-rlcd-200k --encoded-dir prepared/laya_test_tokens --eval-batch-size 64
python merge_predictions.py --input-dir evaluations/sharp-4b --output evaluations/sharp-4b/predictions.jsonl.gz
python merge_predictions.py --input-dir evaluations/sharp-421m-rlcd-200k --output evaluations/sharp-421m-rlcd-200k/predictions.jsonl.gz
python paired_metrics.py --test data/native/en/test.jsonl.gz --baseline-predictions evaluations/sharp-4b/predictions.jsonl.gz --model-predictions evaluations/sharp-421m-rlcd-200k/predictions.jsonl.gz --baseline-name sharp-4b --model-name sharp-421m-rlcd-200k --output evaluations/paired.json
```

For original Laya's reference, use `--checkpoint models/laya-original`; it is an
evaluation only, with zero additional downstream training. For Qwen Head-only,
use `--method head_only --batch-size 512`; the two frozen 4B heads may reuse an
explicit `--cache-dir`, while each distinct LoRA backbone needs its own forward.
For 9B, supply its upstream path and `--model-name Qwen3.5-9B`. A saved training
checkpoint contains the selection/resume metadata; an inference-only exported
Qwen head/adapter can instead be loaded with the model card's inference API.

Accuracy weights test rows equally. Question macro-F1 first averages F1 across
**all legal candidates of each question**, then averages question versions
equally. Invalid model outputs remain incorrect in the full denominator. Paired
intervals use 1,000 bootstrap resamples of the same household/respondent groups
for both models. Complete final-test comparisons require all 240,458 rows;
`--allow-partial` is explicitly labeled and is not used for the paper's final
tables. Apply the paired CLI to RLCD−SFT at both budgets, RLCD200−CEcontinued200,
CEcontinued200−SFT200, and SFT at both budgets−original-reference.

CPU fixture checks:

```sh
python -m unittest discover -s tests -p test_protocol.py -v
python run_recipe.py --recipe sharp-4b-lora-50k --data prepared/qwen --model models/qwen4b-base --output runs/example --dry-run
```
