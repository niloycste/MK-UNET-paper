# Handover — running the sweep and reading the results

Everything here works without further help. Read the first two sections; the rest is
reference for when something looks wrong.

---

## 1. Checking progress

In the machine folder, **double-click `CHECK.bat`** (or, in Anaconda Prompt with
`conda activate mkunetenv`, run `python -W ignore "path A/15_status.py"`).

It prints how many runs are done, each unit's four arms, the pruned-vs-fixed
comparison the paper turns on, and whether anything is currently training. It only
reads files, so it is safe to run at any time, as often as you like.

The line to watch at the bottom:

- `IN PROGRESS: ... (last update 3 min ago)` — healthy.
- `NO PROGRESS FOR OVER AN HOUR` — it has stalled. See section 4.
- `Nothing is training right now` + runs remaining — start `RUN.bat` again.
- `ALL RUNS COMPLETE` — done; copy `path A/runs` to the main machine.

## 2. Starting and restarting

    conda activate mkunetenv
    cd <machine folder>
    RUN.bat

**`RUN.bat` is always safe to re-run.** Finished runs are skipped; an interrupted one
continues from its last completed epoch with optimizer, learning-rate schedule and RNG
state restored. Nothing is ever lost beyond the epoch in progress. If you are unsure
whether it is running, run `CHECK.bat`; if nothing is training, start `RUN.bat`.

Keep the window open — closing it stops training. Minimise instead.

Never run two copies on one machine. Each peaks near 7 GB of RAM and two at once
caused an out-of-memory failure during this project.

---

## 3. Reading the results

Each run writes `path A/runs/<run_id>/result.json`. The number that matters is
**`test_dice_at_best_val`** — test Dice at the epoch with the best validation Dice.
`CHECK.bat` reads these for you.

Every unit is one (model size, dataset) pair trained four ways:

| arm | what it is |
|---|---|
| `fixed [1,3,5]` | the original MK-UNet block — the control |
| `pruned [3,5]` | kernel 1 removed — **the paper's proposal** |
| `adaptive` | learned per-input kernel weights — the search tool |
| `sparse top-1` | one kernel per input — expected to lose |

**The central question is whether `pruned [3,5]` matches or beats `fixed [1,3,5]`
across datasets.** `CHECK.bat` prints that comparison directly.

As of handover it is genuinely mixed — EM `+0.13`, ColonDB `-4.57`. Both outcomes are
publishable: if pruning wins broadly it is a method paper; if it does not, the result
is that conditional computation fails in ultra-lightweight models, supported by the
measurements in `path A/results/branch_analysis.json` (branch cosine 0.389, top-1
discards 63.5% of the signal). **Do not discard runs that came out unfavourably.**

A caveat to carry into the writing: this is **one seed**. Differences smaller than
about 2 Dice points cannot be distinguished from seed noise. Efficiency numbers
(parameters, FLOPs, latency) are deterministic and need no seeds.

## 4. If something looks wrong

**Nothing training, runs remaining** — run `RUN.bat`. This is the normal fix and
covers most cases.

**`ModuleNotFoundError`** — the wrong Python. Run `conda activate mkunetenv` first.
Verify with `python -c "import numpy, torch"` (silence means success).

**`SKIP <unit>: ... not prepared`** — that dataset folder is missing. Copy
`data/<task>/target/<Dataset>/` from the main machine. The run is skipped, not failed;
rerun `RUN.bat` once the data is in place.

**`SKIP sparse: no adaptive run`** — the sparse arm warm-starts from the adaptive arm
of the same unit, so the adaptive one must finish first. Rerunning `RUN.bat` fixes it.

**Verifying the install** — `python -W ignore "path A/02_smoke_test.py"` must report
`24 passed, 0 failed`. Run it on any new machine before starting.

**Machine A only:** it runs under Windows Task Scheduler (`MKUNetMachineA`) with a
watchdog that restarts the sweep every 2 hours if it has died, and again at logon
after a reboot. On machines B and C simply use `RUN.bat` in an open Anaconda Prompt —
simpler, and it avoids the Task Scheduler 72-hour execution limit that would otherwise
kill a multi-week run.

## 5. When a machine finishes

Copy that machine's **`path A/runs/`** folder back to the main machine, into
`C:\MK-UNet\path A\runs\`. That folder is the entire output — `result.json`,
`history.json` (per-epoch curves) and `best.pth` for every run.

With all three merged, `10_collect_results.py` and `11_finalize.py` build the summary
tables, and `03_measure_cost.py`, `04_verify_sparse_execution.py` and
`12_variant_scaling.py` produce the efficiency figures (hours, not days — they do no
training).

## 6. Work plan

| unit | model | dataset |
|---|---|---|
| machineA | base, T, base | EM, ColonDB, ISIC18 |
| machineB | S, base, T | ISIC18, BUSI/ColonDB, DSB18/EM |
| machineC | base, S, T | ClinicDB, BUSI/ColonDB/DSB18, ISIC18/EM |

18 units × 4 arms = **72 runs**, 200 epochs each, roughly 39 days per machine.
`base/ISIC18` on machine A is a single ~37-day job and runs last.
