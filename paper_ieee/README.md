# IEEE Paper Workspace

This folder contains the manuscript source and publication figures for the
MK-UNet-CC extension study. It is separate from the original MK-UNet paper.

- `main.tex`: IEEE double-column manuscript reporting the 72-run, single-seed
  multi-dataset sweep.
- `references.bib`: bibliography used by the draft.
- `make_results_figure.py`: rebuilds the cross-scale Dice-delta figure from the
  generated machine-package run records.
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

Per-machine run logs, command/environment records, and checkpoints are local
artifacts and are intentionally not committed. The checked-in manuscript tables
and PDF figures are the preserved summary snapshot; rebuilding every result
figure requires the prepared data and the generated run records. Every reported
accuracy is a single-seed point estimate. Review author and submission metadata
before using the draft externally.
