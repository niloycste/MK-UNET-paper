# IEEE Paper Workspace

This folder contains the manuscript workspace for the MK-UNet-CC study.

- `main.tex`: IEEE double-column manuscript based on all 72 completed sweep runs.
- `references.bib`: bibliography used by the draft.
- `make_results_figure.py`: rebuilds the cross-scale Dice-delta figure directly
  from completed `dist/machine*/path A/runs/*/result.json` artifacts.
- `make_dataset_figure.py`: rebuilds the six-dataset image/mask overview from
  deterministic median-foreground examples in the prepared test splits.
- `make_method_figures.py`: rebuilds the simplified conditional-routing
  architecture and the separate diagnostic static-ablation figure.

Build from this folder (the verified draft is seven IEEE pages, including references):

```powershell
lualatex main.tex
bibtex main
lualatex main.tex
lualatex main.tex
```

Refresh the generated result figure from the repository root with:

```powershell
python paper_ieee/make_results_figure.py
python paper_ieee/make_dataset_figure.py
python paper_ieee/make_method_figures.py
```

The manuscript includes all verified Machine A/B/C results. Every reported
accuracy remains a single-seed point estimate, and the author affiliation,
email, and acknowledgment are intentionally left as submission placeholders.
