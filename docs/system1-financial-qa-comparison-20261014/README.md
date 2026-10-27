# Decision-model evaluation

[Paper and reported aggregates](../../paper/) · [Training and evaluation code](../../reproduction/training/README.md) · [Model releases and attribution](MODEL_SOURCES.md)

The evaluation interface accepts an explicit state, an authored question and a finite set of candidate options. Accuracy and question-macro F1 summarize the reported model comparisons. [METHODS.md](METHODS.md) defines these metrics; [REPRODUCTION.md](REPRODUCTION.md) links the generic scientific tools and the independent accounting fixture.

Table 4a evaluates fourteen models and a question-majority baseline on 240,458 records and 119 question versions. Table 4b evaluates sharp-421m-mlx and GPT-6-Astra on the same fixed 500-group sample of 14,441 records. [Full-test aggregates](../../paper/source_data/SYSTEM1_model_metrics.csv), [Astra sample aggregates](../../paper/source_data/SYSTEM1_astra_fixed_sample.csv) and [benchmark scope](../../paper/source_data/SYSTEM1_benchmark_scope.json) report the two denominators.
