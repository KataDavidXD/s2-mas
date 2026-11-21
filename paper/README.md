# S²-MAS paper and reported resources

[arXiv](https://arxiv.org/abs/2610.27746) · [Complete manuscript PDF](S2_MAS_Paper.pdf) · [Source ZIP](https://github.com/KataDavidXD/s2-mas/releases/download/paper-2026-11-21/S2_MAS_arXiv_2026-11-21_source.zip) · [LaTeX](main.tex) · [Household experiment source](household_experiment.tex) · [Figures](figures/) · [Reported aggregates](source_data/) · [Plotting scripts](figure_code/)

Manuscript revision, 21 November 2026, 55 pages, 13 main figures and 8 supplementary figures. The YN names identify the compact encoder and 4B/9B decoder decision policies; training suffixes distinguish fixed-budget variants. The household experiment uses the head-only 200k 4B variant, whereas the selected main benchmark 4B uses LoRA plus its decision head.

The paper separates the 500M persona resource, completed 12M-household one-shot application and controlled state-writeback benchmarks. Section 7 adds standalone household consumption forecasts, fully rational and advised policies, income/wealth group MAE, information-timing tests and paired policy responses. Figures 10–13 and Tables 7–10 present these results. The repeated-decision sandbox remains supplementary.

The source package contains main.tex, household_experiment.tex, 00README.json and 21 figure PDFs. Compile main.tex twice with XeLaTeX using TeX Gyre Termes and TeX Gyre Heros. Figure files contain only display content and required rendering resources.

The public scripts cover aggregate figure reproduction, decision-model training/evaluation interfaces and independent synthetic companion generation. Household records and individual trajectories remain restricted; the GENESE/MAS engine implementation remains private. Component terms are linked in the repository README.

The 21 November revision redraws Figures 5 and 8, reduces repeated columns in the task and prediction tables, reports prediction scores to three decimal places, and incorporates language and cross-reference edits. The former component-summary Table 5 has been removed. Training samples, exposures and costs are reported in Supplementary Table S24.

To reproduce the revised figures from the aggregate data, run from the paper directory with Python and Matplotlib installed.

    python figure_code/financial_training.py --data-dir source_data --output-dir figures
    python figure_code/scale12m.py --data-dir source_data --output-dir figures
