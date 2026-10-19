# Choice evaluation methods

Each input supplies its candidate IDs and one expected candidate. The evaluator preserves candidate ordering and counts invalid answers separately while retaining the complete evaluation denominator.

Accuracy is the number of correct choices divided by all evaluation rows. For each question, macro F1 averages over its complete legal candidate set with zero division assigned zero. Question-macro F1 then gives each question equal weight. A paired comparison uses the same rows, candidate sets and grouping units for both models.

Checkpoint and method selection use development data. The final test evaluates the fixed selected versions. Unique training inputs and repeated row exposures describe different budgets and are recorded separately in the reported model aggregates.

[Scientific evaluation tools](../../reproduction/training/README.md) accept generic task and question IDs. [Model releases and attribution](MODEL_SOURCES.md) preserve the upstream implementation and checkpoint identities.
