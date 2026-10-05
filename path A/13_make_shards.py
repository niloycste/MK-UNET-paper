"""Split the full experiment grid across several machines, balanced by cost.

The naive split -- one variant per machine -- is badly lopsided: base costs 597 h
against T's 122 h, so one machine would finish in 5 days and another in 25. This
packs individual (variant, dataset) units into shards of roughly equal predicted
runtime instead, then writes one runnable script per machine.

Cost model is extrapolated from MEASURED throughput on the reference machine
(i9-14900, 24 cores): ClinicDB / MK_UNet_T / fixed / 352px / 489 train images =
3.2 min per epoch. Scaling is linear in image count and in model GFLOPs, quadratic
in resolution. Predictions are therefore only valid for machines of similar speed --
see --speed.

    python -W ignore "path A/13_make_shards.py" --shards 3
    python -W ignore "path A/13_make_shards.py" --shards 3 --variants T S --seeds 1
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'shards')

# dataset -> (train images, resolution, data_root)
DATASETS = {
    'EM':       (24,   256, 'data/cell/target'),
    'ColonDB':  (303,  352, 'data/polyp/target'),
    'ClinicDB': (489,  352, 'data/polyp/target'),
    'BUSI':     (518,  256, 'data/busi/target'),
    'DSB18':    (536,  256, 'data/cell/target'),
    'ISIC18':   (2075, 256, 'data/isic/target'),
}
GFLOPS = {'T': 0.0668, 'S': 0.1323, 'base': 0.3275, 'M': 0.9770, 'L': 3.2451}
NETNAME = {'T': 'MK_UNet_T', 'S': 'MK_UNet_S', 'base': 'MK_UNet',
           'M': 'MK_UNet_M', 'L': 'MK_UNet_L'}

# reference measurement
REF_MIN_PER_EPOCH, REF_IMGS, REF_PX, REF_GF = 3.2, 489, 352, GFLOPS['T']

# Resolution scaling exponent, FITTED to measurement rather than assumed.
#
# The first version used 2.0 on the reasoning that convolution cost is quadratic in
# side length. Measured on this machine:
#     ClinicDB/T/352px/489 imgs -> 3.20 min/epoch  (0.00654 min per image)
#     BUSI    /T/256px/518 imgs -> 3.00 min/epoch  (0.00579 min per image)
# so going 352 -> 256 costs 0.885x, not the 0.529x an exponent of 2 predicts:
#     ln(0.885) / ln(256/352) = 0.38
# At these model sizes the per-image overhead -- JPEG decode, augmentation, and the
# val+test evaluation run every epoch -- dominates the convolutions, so wall-clock is
# far flatter in resolution than FLOPs are. Using 2.0 understated every 256px dataset
# by 1.68x, and ISIC18 (256px) is the single largest cost in the sweep.
RES_EXPONENT = 0.38

# arm -> (cost multiplier vs a fixed run of the same length, extra CLI args)
# adaptive is 1.63x because of the per-sample weighted sum (measured 5.2 vs 3.2 min/ep);
# sparse is warm-started so it runs SPARSE_EPOCH_FRAC of the budget: 1.63 * 0.6.
ARMS = {
    'fixed':    (1.00, ['--routing_mode', 'fixed', '--kernel_sizes', '1', '3', '5']),
    'pruned':   (0.93, ['--routing_mode', 'fixed', '--kernel_sizes', '3', '5']),
    'adaptive': (1.63, ['--routing_mode', 'adaptive_soft']),
    'sparse':   (0.98, ['--routing_mode', 'sparse_top1', '--lr', '0.0003']),
}
# The sparse arm starts from the adaptive checkpoint, so it needs only a fraction of
# the budget. These were 60 and 20 when --epochs was 100; expressing them as fractions
# keeps the schedule proportional when the budget changes, and keeps the 0.98 cost
# multiplier above valid at any --epochs.
SPARSE_EPOCH_FRAC, SPARSE_ANNEAL_FRAC = 0.6, 0.2


def hours(variant, dataset, epochs):
    n, px, _ = DATASETS[dataset]
    m = (REF_MIN_PER_EPOCH * (GFLOPS[variant] / REF_GF) * (n / REF_IMGS)
         * (px / REF_PX) ** RES_EXPONENT)
    return m * epochs / 60


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shards', type=int, default=3, help='number of machines')
    ap.add_argument('--variants', nargs='+', default=['T', 'S', 'base'])
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    ap.add_argument('--arms', nargs='+', default=list(ARMS))
    ap.add_argument('--seeds', type=int, default=1)
    ap.add_argument('--seed0', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=200,
                    help='the paper trains for 200. At 100 every run was still '
                         'improving, which confounds arm comparisons with '
                         'convergence rate, so 200 is the default here.')
    ap.add_argument('--speed', type=float, nargs='*', default=None,
                    help='relative speed per machine (1.0 = reference i9-14900). '
                         'e.g. --speed 1.0 0.6 0.8 for a slower second machine')
    a = ap.parse_args()

    speed = a.speed or [1.0] * a.shards
    if len(speed) != a.shards:
        sys.exit(f'--speed needs {a.shards} values, got {len(speed)}')

    # One unit = one (variant, dataset, seed): all arms for it stay together, since
    # sparse warm-starts from the adaptive run of the same variant/dataset/seed.
    units = []
    for v in a.variants:
        for d in a.datasets:
            for s in range(a.seeds):
                cost = sum(ARMS[arm][0] for arm in a.arms) * hours(v, d, a.epochs)
                units.append({'variant': v, 'dataset': d, 'seed': a.seed0 + s,
                              'hours': cost})
    units.sort(key=lambda u: -u['hours'])            # largest first (greedy packing)

    # weight each bin by machine speed so a slower machine gets less work
    bins = [{'id': i + 1, 'speed': speed[i], 'units': [], 'load': 0.0}
            for i in range(a.shards)]
    for u in units:
        b = min(bins, key=lambda b: (b['load'] + u['hours']) / b['speed'])
        b['units'].append(u)
        b['load'] += u['hours']

    os.makedirs(OUT, exist_ok=True)
    total = sum(u['hours'] for u in units)
    print(f'{len(units)} work units, {total:.0f} h total on the reference machine\n')

    for b in bins:
        wall = b['load'] / b['speed']
        print(f"machine {b['id']}  (speed {b['speed']}x): {len(b['units']):3d} units, "
              f"{b['load']:6.0f} ref-h -> {wall:6.0f} h wall = {wall/24:.1f} days")
        lines = [
            '# Auto-generated by 13_make_shards.py -- do not edit by hand.',
            f"# Machine {b['id']} of {a.shards}. Predicted wall time {wall/24:.1f} days.",
            '#',
            '# Runs strictly sequentially: two concurrent jobs peak ~7 GB RSS each and',
            '# caused an out-of-memory crash earlier in this project.',
            '#',
            '# Arms for one (variant, dataset, seed) are kept on the SAME machine because',
            '# the sparse arm warm-starts from that unit\'s adaptive checkpoint.',
            '#',
            '# Safe to re-run: every call passes --resume, so finished units skip and an',
            '# interrupted one continues from its last completed epoch.',
            '',
            '# Paths are relative to this script, so the folder can live anywhere.',
            '$here = Split-Path -Parent $MyInvocation.MyCommand.Definition',
            'Set-Location $here',
            '',
            '# Uses "python" from PATH (activate the conda env first). Override with:',
            '#   $env:MKUNET_PY = "C:\\path\\to\\python.exe"',
            '$py = if ($env:MKUNET_PY) { $env:MKUNET_PY } else { "python" }',
            '',
            'if (-not (Get-Command $py -ErrorAction SilentlyContinue)) {',
            '    Write-Output "ERROR: python not found. Activate the conda environment, or set '
            '$env:MKUNET_PY to the interpreter path."; exit 1',
            '}',
            '',
        ]
        # Balance is decided above; this only fixes the ORDER of execution. Cheap units
        # run first so a broken shard shows up in hours, and ISIC18 runs last because
        # it is the one dataset that may still be downloading when a machine starts.
        for u in sorted(b['units'], key=lambda u: (u['dataset'] == 'ISIC18', u['hours'])):
            net, ds, seed = NETNAME[u['variant']], u['dataset'], u['seed']
            _, px, root = DATASETS[ds]
            tag = f"{u['variant']}/{ds}/seed{seed}"
            lines += [
                f'Write-Output "=== {tag}   [$(Get-Date -Format \'MM-dd HH:mm\')] ==="',
                f'if (-not (Test-Path "{root}/{ds}/train/images")) {{',
                f'  Write-Output "  SKIP {tag}: {root}/{ds} not prepared"',
                '} else {',
            ]
            common = ['--network', net, '--dataset', ds, '--data_root', root,
                      '--img_size', str(px), '--seed', str(seed),
                      '--runs', '1', '--device', 'cpu', '--aux_supervision', 'False',
                      '--resume']
            for arm in a.arms:
                _, extra = ARMS[arm]
                if arm == 'sparse':
                    # The adaptive run_id carries a timestamp that is unknown until it
                    # finishes, so resolve its checkpoint at runtime rather than
                    # hard-coding a path that cannot exist yet.
                    pat = f'{ds}_{net}_adaptive_soft_k135_auxFalse_e{a.epochs}_seed{seed}_t*'
                    lines += [
                        f'  $src = Get-ChildItem "path A\\runs" -Directory '
                        f'-Filter "{pat}" -ErrorAction SilentlyContinue | '
                        f'Sort-Object Name | Select-Object -Last 1',
                        '  if (-not $src) { Write-Output "  SKIP sparse: no adaptive run" }',
                        '  else {',
                        '    $ck = Join-Path $src.FullName "best.pth"',
                        '    & $py -W ignore "path A\\05_train.py" '
                        + ' '.join(extra + common + [
                            '--epoch', str(int(SPARSE_EPOCH_FRAC * a.epochs)),
                            '--anneal_epochs', str(int(SPARSE_ANNEAL_FRAC * a.epochs))])
                        + ' --init_from "$ck"',
                        '  }',
                    ]
                else:
                    lines.append('  & $py -W ignore "path A\\05_train.py" '
                                 + ' '.join(extra + common
                                            + ['--epoch', str(a.epochs), '--lr', '0.0005']))
            lines += ['}', '']
        lines.append(f'Write-Output "Machine {b["id"]} complete."')
        p = os.path.join(OUT, f'machine{b["id"]}.ps1')
        open(p, 'w', encoding='utf-8').write('\n'.join(lines))

    json.dump({'shards': a.shards, 'epochs': a.epochs, 'seeds': a.seeds,
               'total_ref_hours': total,
               'bins': [{'id': b['id'], 'speed': b['speed'], 'ref_hours': b['load'],
                         'wall_hours': b['load'] / b['speed'],
                         'units': b['units']} for b in bins]},
              open(os.path.join(OUT, 'plan.json'), 'w'), indent=2)
    print(f'\nwrote {OUT}\\machine*.ps1 and plan.json')
    print('\nNOTE: predictions assume machines as fast as the i9-14900 reference.')
    print('Pass --speed to rebalance, e.g.  --speed 1.0 0.6 0.8')


if __name__ == '__main__':
    main()
