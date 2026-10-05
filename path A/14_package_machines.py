"""Build one self-contained folder per machine, ready to copy and run.

13_make_shards.py decides WHICH runs go where; this assembles the folders that make
those runs possible on a machine that has never seen the project. Each folder holds
the code, only the datasets its own shard needs, a portable run.ps1, and a README.

Copying the whole repo to every machine would also work, but it ships the raw
downloads (~3.3 GB, and none of it is read at training time) and leaves the operator
to figure out which script to run. This ships ~200-500 MB and one command.

Run 13_make_shards.py first, then:

    python -W ignore "path A/14_package_machines.py"
    python -W ignore "path A/14_package_machines.py" --out D:/to_copy --machines 2 3
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHARDS = os.path.join(HERE, 'shards')

# dataset -> prepared data directory, relative to the repo root
DATA_DIR = {
    'EM':       'data/cell/target/EM',
    'DSB18':    'data/cell/target/DSB18',
    'ColonDB':  'data/polyp/target/ColonDB',
    'ClinicDB': 'data/polyp/target/ClinicDB',
    'BUSI':     'data/busi/target/BUSI',
    'ISIC18':   'data/isic/target/ISIC18',
}

# Everything a training run imports, plus the tools needed to evaluate and to
# re-prepare a dataset that is missing. Deliberately excludes data/*/raw downloads.
CODE_FILES = ['mkunet_network.py', 'train_polyp.py', 'test_polyp.py',
              'prepare_datasets.py', 'requirements.txt', 'LICENSE']
CODE_DIRS = ['utils']
EXT_FILES = ['routing.py', '05_train.py', '06_evaluate.py', '07_reliability.py',
             '08_analyze_branches.py', '09_kernel_subsets.py', '10_collect_results.py',
             '11_finalize.py', '12_variant_scaling.py', '03_measure_cost.py',
             '04_verify_sparse_execution.py', '02_smoke_test.py', '15_status.py']

README = """# Machine {letter} -- MK-UNet extension sweep

Self-contained. Copy this whole folder to the machine and run it; nothing else
from the original project is needed.

## Workload

{units} work units, {arms} arms each, {epochs} epochs.
Predicted wall time: **{days:.1f} days** on an i9-14900 (24 cores).

{table}

## Setup (once)

1. Install Miniconda, then:

       conda create -n mkunetenv python=3.9 -y
       conda activate mkunetenv
       pip install -r requirements.txt

   `requirements.txt` pins the CUDA build of torch. This sweep is CPU-only, so the
   CPU wheel is fine and much smaller:

       pip install torch==1.11.0 torchvision==0.12.0

2. Check the environment before committing days to it:

       python -W ignore "path A/02_smoke_test.py"

   24 checks; all must pass.

## Run

Open **Anaconda Prompt**, then:

    conda activate mkunetenv
    cd <path to this folder>
    RUN.bat

Or just **double-click `RUN.bat`** if `mkunetenv` is your default environment.

Leave the window open -- closing it stops the run. Progress is written to
`run.log` as well as the window.

**If the machine reboots or the run dies, start it again the same way.** Every
call passes `--resume`: finished units are skipped and an interrupted one picks up
from its last completed epoch, with optimizer, LR schedule and RNG state restored.

Python not on PATH? Point at it directly:

    $env:MKUNET_PY = "C:\\Users\\you\\.conda\\envs\\mkunetenv\\python.exe"

## Results

One directory per run under `path A/runs/`, each with `result.json`
(`test_dice_at_best_val`), `history.json` (per-epoch curve), `best.pth`, and
`config.json`. **Copy `path A/runs/` back to the main machine when finished** --
that is the entire output.

## Notes

- Runs strictly one at a time on purpose. Two concurrent jobs peak around 7 GB RSS
  each and have already caused one out-of-memory failure. Do not start a second copy.
- {isic}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(ROOT, 'dist'),
                    help='where to build the folders (default: <repo>/dist)')
    ap.add_argument('--machines', nargs='*', type=int, default=None,
                    help='machine numbers to build (default: all in plan.json)')
    ap.add_argument('--no_data', action='store_true',
                    help='copy code only, skip the datasets')
    a = ap.parse_args()

    plan_path = os.path.join(SHARDS, 'plan.json')
    if not os.path.exists(plan_path):
        sys.exit('shards/plan.json not found -- run 13_make_shards.py first')
    plan = json.load(open(plan_path))

    for b in plan['bins']:
        if a.machines and b['id'] not in a.machines:
            continue
        letter = chr(ord('A') + b['id'] - 1)
        dest = os.path.join(a.out, f'machine{letter}')

        # Rebuilding replaces code and data, but must never destroy completed training.
        # A rebuild can happen weeks into a sweep (the cost model was recalibrated
        # mid-run once already), so any existing runs/ is set aside and restored.
        stash = None
        runs = os.path.join(dest, 'path A', 'runs')
        if os.path.isdir(runs):
            stash = os.path.join(a.out, f'.runs_{letter}')
            if os.path.isdir(stash):
                shutil.rmtree(stash)
            shutil.move(runs, stash)
        if os.path.isdir(dest):
            shutil.rmtree(dest)                 # rebuilding must not merge stale output
        os.makedirs(dest)

        for f in CODE_FILES:
            src = os.path.join(ROOT, f)
            if os.path.exists(src):
                shutil.copy2(src, dest)
        for d in CODE_DIRS:
            shutil.copytree(os.path.join(ROOT, d), os.path.join(dest, d),
                            ignore=shutil.ignore_patterns('__pycache__'))

        ext = os.path.join(dest, 'path A')
        os.makedirs(ext)
        for f in EXT_FILES:
            src = os.path.join(HERE, f)
            if os.path.exists(src):
                shutil.copy2(src, ext)

        if stash:
            shutil.move(stash, runs)
            print(f'  machine{letter}: preserved {len(os.listdir(runs))} existing run(s)')

        shutil.copy2(os.path.join(SHARDS, f'machine{b["id"]}.ps1'),
                     os.path.join(dest, 'run.ps1'))

        # Read-only progress report. Double-clickable so checking on the sweep never
        # requires remembering a command.
        open(os.path.join(dest, 'CHECK.bat'), 'w', encoding='utf-8', newline='').write(
            '@echo off\r\n'
            'cd /d "%~dp0"\r\n'
            'where python >nul 2>&1\r\n'
            'if errorlevel 1 (\r\n'
            '  echo Open "Anaconda Prompt", type:  conda activate mkunetenv\r\n'
            '  echo then run this file again from there.\r\n'
            '  pause\r\n'
            '  exit /b 1\r\n'
            ')\r\n'
            'python -W ignore "path A\\15_status.py"\r\n'
            'echo.\r\n'
            'pause\r\n')

        # Double-clickable entry point. run.ps1 alone needs the right execution policy
        # and an activated environment; this checks both and says what to do if not.
        # newline='' so the explicit \r\n below is written verbatim; text mode would
        # translate each \n again and double-space the file.
        open(os.path.join(dest, 'RUN.bat'), 'w', encoding='utf-8', newline='').write(
            '@echo off\r\n'
            'cd /d "%~dp0"\r\n'
            'where python >nul 2>&1\r\n'
            'if errorlevel 1 (\r\n'
            '  echo.\r\n'
            '  echo Python was not found.\r\n'
            '  echo Open "Anaconda Prompt", type:  conda activate mkunetenv\r\n'
            '  echo then run this file again from there.\r\n'
            '  echo.\r\n'
            '  pause\r\n'
            '  exit /b 1\r\n'
            ')\r\n'
            'echo Starting. This runs for days. Closing this window stops it.\r\n'
            'echo Progress is written to run.log as well as this window.\r\n'
            'echo.\r\n'
            # One powershell with Tee-Object inside it. Piping between two powershell
            # processes also works but is needlessly fragile.
            'powershell -NoProfile -ExecutionPolicy Bypass -Command '
            '"& \'%~dp0run.ps1\' *>&1 | Tee-Object -FilePath \'%~dp0run.log\' -Append"\r\n'
            'echo.\r\n'
            'echo Finished. Copy the "path A\\runs" folder back to the main machine.\r\n'
            'pause\r\n')

        # Only this machine's datasets, so a folder stays small enough to move on a stick.
        wanted = sorted({u['dataset'] for u in b['units']})
        missing, copied_mb = [], 0.0
        if not a.no_data:
            for ds in wanted:
                src = os.path.join(ROOT, DATA_DIR[ds])
                if not os.path.isdir(src):
                    missing.append(ds)
                    continue
                dst = os.path.join(dest, DATA_DIR[ds].replace('/', os.sep))
                shutil.copytree(src, dst)
                copied_mb += sum(os.path.getsize(os.path.join(r, f))
                                 for r, _, fs in os.walk(dst) for f in fs) / 1e6

        rows = sorted(b['units'], key=lambda u: -u['hours'])
        table = ('| variant | dataset | hours |\n|---|---|---:|\n'
                 + '\n'.join(f"| {u['variant']} | {u['dataset']} | {u['hours']:.0f} |"
                             for u in rows))
        isic = ('ISIC18 is included and prepared.' if 'ISIC18' in wanted
                and 'ISIC18' not in missing else
                'ISIC18 is NOT in this shard.' if 'ISIC18' not in wanted else
                'ISIC18 is still downloading and is NOT in this folder. It is scheduled '
                'last, so there is time: prepare it on the main machine, then copy\n  '
                '`data/isic/target/ISIC18/` into the same path here before the run '
                'reaches it. If it is still missing, that unit is skipped with a message '
                'rather than crashing.')

        open(os.path.join(dest, 'README.md'), 'w', encoding='utf-8').write(
            README.format(letter=letter, units=len(b['units']),
                          arms=len(plan.get('arms', [1, 2, 3, 4])) or 4,
                          epochs=plan['epochs'], days=b['wall_hours'] / 24,
                          table=table, isic=isic))

        size = sum(os.path.getsize(os.path.join(r, f))
                   for r, _, fs in os.walk(dest) for f in fs) / 1e6
        print(f'machine{letter}: {len(b["units"])} units, {b["wall_hours"]/24:4.1f} days, '
              f'{size:7.1f} MB  ({", ".join(wanted)})')
        if missing:
            print(f'            NOT COPIED (not prepared yet): {", ".join(missing)}')

    print(f'\nbuilt in {a.out}')
    print('Copy a machine<X> folder to that machine, then: '
          'conda activate mkunetenv; powershell -ExecutionPolicy Bypass -File run.ps1')


if __name__ == '__main__':
    main()
