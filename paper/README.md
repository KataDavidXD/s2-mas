# S²-MAS paper and reported resources

[Latest manuscript PDF](S2_MAS_Paper.pdf) · [LaTeX](main.tex) · [Figures](figures/) · [Reported aggregates](source_data/) · [Plotting scripts](figure_code/) · [Poster](posters/S2_MAS_Poster.pdf) · [Social poster](posters/S2_MAS_Poster_Social.pdf)

The paper connects a 500M-persona resource, six decision-model routes and ten trained release versions with household choice and financial state evolution. The 12M-household application reports age/income and financial-stock heterogeneity alongside 18 additional persona attributes. Figure 7 and Tables S41–S47 connect consumption responses to purchase choices, cash use and actual payments. Tables S46–S47 give the education, employment and household-size comparisons and the complete category results. Figures 8–10 cover the five-shock design, shock-relative adjustments and within-route purchase-goal contrasts. Appendix S13 retains the two-million-household fixed-plan credit application.

[Synthetic companion](../resources/public_companion/README.md), [CC0 Wiki context](../resources/wikidata/README.md) and the [model catalog](../resources/MODEL_CATALOG.md) have separate component entries and terms. Public code covers scientific plotting, decision-model training/evaluation and independent synthetic companion generation. The GENESE/MAS engine implementation remains private.

Compile `main.tex` twice with XeLaTeX using TeX Gyre Termes and TeX Gyre Heros. Replot figures from the supplied aggregate CSVs:

```sh
python figure_code/scale12m_heterogeneity.py --data-dir source_data --output-dir artifacts/heterogeneity
python figure_code/replot_dynamic_shocks.py --source-data source_data --output-dir artifacts/shock-figures
python figure_code/framework_overview.py --output-dir figures --qa-dir artifacts/figure-layout
python figure_code/decision_training_routes.py --output-dir figures --qa-dir artifacts/figure-layout
```

The remaining plotting scripts use their associated aggregate tables. Code and data components retain the terms supplied at their entry points.
