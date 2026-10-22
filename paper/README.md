# S²-MAS paper and reported resources

[arXiv](https://arxiv.org/abs/2610.27746) · [Latest manuscript PDF](S2_MAS_Paper.pdf) · [LaTeX](main.tex) · [Figures](figures/) · [Reported aggregates](source_data/) · [Plotting scripts](figure_code/)

Version v54 contains 59 pages, 13 main figures and 7 supplementary figures. S²-MAS links 500 million household personas, trainable decision models and a stateful execution engine to consumption, cash and repayment paths. sharp-4b and sharp-421m are the principal decision policies; sharp-421m-mlx executes the reported household applications.

Figure 9 and Tables S41–S47 explain consumption heterogeneity in the 12M-household application. Figure 10 defines the five shock paths. Table 8 fixes the purchase goal at CNY 20,000; Figure 11 and Table 9 compare each shock with its same-goal unshocked reference. Figure 12 maps the directions of shock and goal effects, and Figure 13 follows the within-route 30k-minus-20k contrasts through 24 months. Tables S49–S50 support the purchase-timing and bank-offer explanations. Appendix S13 retains the two-million-household fixed-plan credit application.

[Synthetic companion](../resources/public_companion/README.md), [CC0 Wikidata context](../resources/wikidata/README.md) and the [model catalog](../resources/MODEL_CATALOG.md) have separate component entries and terms. Public code covers scientific plotting, decision-model training and evaluation, and independent synthetic companion generation. The GENESE/MAS engine implementation remains private.

Compile `main.tex` twice with XeLaTeX using TeX Gyre Termes and TeX Gyre Heros. Replot the shock direction map from the reported monthly means:

```sh
python figure_code/shock_direction_map.py --monthly-source source_data/dynamic_shock_monthly_means.csv --output-dir artifacts/directions --source-data-dir artifacts/source_data --qa-dir artifacts/figure-layout
```

Other figure scripts use their associated aggregate tables. Code and data retain the terms supplied at their component entries.
