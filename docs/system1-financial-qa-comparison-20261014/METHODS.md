# Choice evaluation methods

Each input supplies its candidate IDs and one expected candidate. The evaluator preserves candidate ordering and counts invalid answers separately while retaining the complete evaluation denominator.

Accuracy is the number of correct choices divided by all evaluation rows. For each question, macro F1 averages over its complete legal candidate set with zero division assigned zero. Question-macro F1 then gives each question equal weight. A paired comparison uses the same rows, candidate sets and grouping units for both models.

The full benchmark contains 240,458 records across 119 question versions and 8,275 household/respondent groups. The GPT-6-Astra comparison uses a fixed 500-group sample of 14,441 records, covering all 119 versions; sharp-421m-mlx is rescored on those same records. GLM-5.3-Flash has 98 invalid responses and DeepSeek-V4.1-Flash has 58, all retained in the denominator.

Checkpoint and method selection use development data. The final test evaluates the fixed selected versions. Unique training inputs and repeated row exposures describe different budgets and are recorded separately in the reported model aggregates.

[Scientific evaluation tools](../../reproduction/training/README.md) accept generic task and question IDs. [Model releases and attribution](MODEL_SOURCES.md) preserve the upstream implementation and checkpoint identities.

The shared 119-version registry was fixed for the initial model comparison and retained for the API extension. The fixed Astra sample was selected before the new outputs by random complete-group sampling with seed 20261020. Its paired interval jointly resamples all records of each selected group for both models, using 1,000 draws and seed 20261018. [Paired interval aggregates](../../paper/source_data/SYSTEM1_astra_paired_interval.csv) preserve the full-precision results.

[Recorded API settings](../../paper/source_data/SYSTEM1_added_api_settings.csv) describe the added GPT-6 and DeepSeek comparisons, including reasoning/thinking settings, output budgets, parsing and transport retry rules. These adapters serialize complete state and question objects without truncation, retaining candidate meanings and order. The prediction targets are recorded questionnaire answers and coded respondent attributes. Authored financial applications use separate action menus and accounting rules.
