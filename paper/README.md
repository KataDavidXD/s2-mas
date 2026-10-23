# S²-MAS paper and reported resources

[arXiv](https://arxiv.org/abs/2610.27746) · [Latest manuscript PDF](S2_MAS_Paper.pdf) · [LaTeX](main.tex) · [Figures](figures/) · [Reported aggregates](source_data/) · [Plotting scripts](figure_code/)

Version v59 contains 50 pages, 9 main figures and 7 supplementary figures. S²-MAS links 500 million household personas, trainable decision models and a stateful execution engine to consumption, cash and repayment paths. sharp-4b and sharp-421m are the principal decision policies; sharp-421m-mlx executes the reported household applications.

Table 4 unifies the one-shot consumer financial-choice prediction benchmark: panel a reports fourteen models and the question-majority baseline on 240,458 records; panel b compares sharp-421m-mlx with GPT-6-Astra on the same 14,441 records from 500 groups. The benchmark measures recorded-answer and coded-target prediction. Model sizes, the added API configurations and the paired Astra interval are reported in [source data](source_data/). The large application demonstrates one-shot scalability through the policy-to-ledger runner.

Figure 9 and Tables S38–S44 explain consumption heterogeneity in the 12M-household application. Appendix S12 retains the two-million-household fixed-plan credit application.

[Synthetic companion](../resources/public_companion/README.md), [CC0 Wikidata context](../resources/wikidata/README.md) and the [model catalog](../resources/MODEL_CATALOG.md) have separate component entries and terms. Public code covers scientific plotting, decision-model training and evaluation, and independent synthetic companion generation. The GENESE/MAS engine implementation remains private.

Compile `main.tex` twice with XeLaTeX using TeX Gyre Termes and TeX Gyre Heros.

Other figure scripts use their associated aggregate tables. Code and data retain the terms supplied at their component entries.
