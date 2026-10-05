"""Show what this machine has finished, what is running, and what is left.

Safe to run at any time; it only reads files. Run it from the machine folder:

    python -W ignore "path A/15_status.py"

or double-click CHECK.bat.
"""
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
ROOT = os.path.dirname(HERE)

ARM = {('fixed', '135'): 'fixed [1,3,5]', ('fixed', '35'): 'pruned [3,5]',
       ('adaptive_soft', '135'): 'adaptive', ('sparse_top1', '135'): 'sparse top-1'}
ORDER = ['fixed [1,3,5]', 'pruned [3,5]', 'adaptive', 'sparse top-1']


def arm_of(cfg):
    ks = ''.join(str(k) for k in cfg['kernel_sizes'])
    return ARM.get((cfg['routing_mode'], ks), cfg['routing_mode'])


def main():
    if not os.path.isdir(RUNS):
        print('No runs yet. If RUN.bat is going, the first epoch takes a few minutes.')
        return

    planned = 0
    rp = os.path.join(ROOT, 'run.ps1')
    if os.path.exists(rp):
        planned = 4 * len(re.findall(r'^Write-Output "=== ', open(rp, encoding='utf-8').read(), re.M))

    done, running, units = [], [], {}
    for name in sorted(os.listdir(RUNS)):
        d = os.path.join(RUNS, name)
        cfile = os.path.join(d, 'config.json')
        if not os.path.exists(cfile):
            continue
        cfg = json.load(open(cfile))
        key = f"{cfg['network'].replace('MK_UNet', 'base').replace('base_', '')}/{cfg['dataset']}"
        row = {'arm': arm_of(cfg), 'unit': key, 'dir': d}
        res = os.path.join(d, 'result.json')
        if os.path.exists(res):
            r = json.load(open(res))
            row['dice'] = r['test_dice_at_best_val']
            row['epoch'] = r['best_epoch']
            done.append(row)
        else:
            hist = os.path.join(d, 'history.json')
            n = len(json.load(open(hist))) if os.path.exists(hist) else 0
            age = (time.time() - os.path.getmtime(hist)) / 60 if os.path.exists(hist) else 999
            row.update(epochs=n, total=cfg['epoch'], stale_min=age)
            running.append(row)
        units.setdefault(key, {})[row['arm']] = row

    print('=' * 66)
    print(f'  FINISHED {len(done)} of {planned or "?"} runs')
    print('=' * 66)

    for unit in sorted(units):
        print(f'\n{unit}')
        for arm in ORDER:
            r = units[unit].get(arm)
            if not r:
                print(f'   {arm:<16} -- not started')
            elif 'dice' in r:
                print(f'   {arm:<16} {r["dice"]:.4f}   (best epoch {r["epoch"]})')
            else:
                print(f'   {arm:<16} RUNNING  epoch {r["epochs"]}/{r["total"]}')

    # The comparison the paper turns on.
    print('\n' + '=' * 66)
    print('  PRUNED [3,5]  vs  FIXED [1,3,5]      <- the key result')
    print('=' * 66)
    any_pair = False
    for unit in sorted(units):
        f = units[unit].get('fixed [1,3,5]', {}).get('dice')
        p = units[unit].get('pruned [3,5]', {}).get('dice')
        if f is not None and p is not None:
            any_pair = True
            d = (p - f) * 100
            verdict = 'pruned WINS' if d > 0 else 'pruned loses'
            print(f'  {unit:<16} fixed {f:.4f}   pruned {p:.4f}   {d:+.2f} pts   {verdict}')
    if not any_pair:
        print('  (no unit has both arms finished yet)')

    print('\n' + '=' * 66)
    if running:
        for r in running:
            warn = '   <-- NO PROGRESS FOR OVER AN HOUR' if r['stale_min'] > 60 else ''
            print(f'  IN PROGRESS: {r["unit"]} {r["arm"]}  epoch {r["epochs"]}/{r["total"]}'
                  f'  (last update {r["stale_min"]:.0f} min ago){warn}')
    else:
        print('  Nothing is training right now.')
        if planned and len(done) >= planned:
            print('  ALL RUNS COMPLETE. Copy the "path A/runs" folder to the main machine.')
        else:
            print('  Not all runs are done -- start RUN.bat again (it resumes).')
    print('=' * 66)


if __name__ == '__main__':
    main()
