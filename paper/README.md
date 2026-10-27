# S²-MAS paper and reported resources

[arXiv](https://arxiv.org/abs/2610.27746) · [Complete manuscript PDF](S2_MAS_Paper.pdf) · [LaTeX](main.tex) · [Household experiment source](household_experiment.tex) · [Figures](figures/) · [Reported aggregates](source_data/) · [Plotting scripts](figure_code/)

Version v67, 28 October 2026: 56 pages, 13 main figures and 8 supplementary figures. The YN names identify the compact encoder and 4B/9B decoder decision policies; training suffixes distinguish fixed-budget variants. The household experiment uses the head-only 200k 4B variant, whereas the selected main benchmark 4B uses LoRA plus its decision head.

The paper separates the 500M persona resource, completed 12M-household one-shot application and controlled state-writeback benchmarks. Section 7 adds standalone household consumption forecasts, fully rational and advised policies, income/wealth group MAE, information-timing tests and paired policy responses. Figures 10–13 and Tables 8–11 present these results. The repeated-decision sandbox remains supplementary.

The source package contains main.tex, household_experiment.tex, 00README.json and 21 figure PDFs. Compile main.tex twice with XeLaTeX using TeX Gyre Termes and TeX Gyre Heros. Figure files contain only display content and required rendering resources.

The public scripts cover aggregate figure reproduction, decision-model training/evaluation interfaces and independent synthetic companion generation. Household records and individual trajectories remain restricted; the GENESE/MAS engine implementation remains private. Component terms are linked in the repository README.
