"""Does sparse routing still save computation as the model gets bigger?

The saving comes from skipping depth-wise branches. In an MKIR block with channel
width C and expansion 2, the branch FLOPs are 70*C*HW against 4*C^2*HW for the two
point-wise convolutions, so the skippable share is

    70C / (4C^2 + 70C)   ->   large when C is small, negligible when C is large.

That means the benefit is a property of the REGIME, not of the method. This script
measures it for every MK-UNet variant instead of assuming it.

Run: python -W ignore "path A/12_variant_scaling.py"
"""
import json
import os
import sys

import torch
from thop import profile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routing import build_model, NET_CONFIGS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
os.makedirs(RESULTS, exist_ok=True)

x = torch.randn(1, 3, 256, 256)
rows = []

print(f'{"variant":<12}{"channels":<26}{"fixed params":>13}{"fixed GFLOPs":>14}'
      f'{"sparse GFLOPs":>15}{"FLOP saving":>13}{"theory":>9}')
print('-' * 103)

for name, ch in NET_CONFIGS.items():
    m_fix = build_model('fixed', name).eval()
    m_spa = build_model('sparse_top1', name).eval()
    p_fix = sum(q.numel() for q in m_fix.parameters())
    p_spa = sum(q.numel() for q in m_spa.parameters())
    f_fix, _ = profile(m_fix, inputs=(x,), verbose=False)
    f_spa, _ = profile(m_spa, inputs=(x,), verbose=False)
    saving = 100 * (f_fix - f_spa) / f_fix

    # analytic skippable share, averaged over the block widths this variant uses
    shares = [70 * c / (4 * c * c + 70 * c) for c in ch]
    theory = 100 * sum(shares) / len(shares) * (2 / 3)   # top-1 drops 2 of 3 branches

    print(f'{name:<12}{str(ch):<26}{p_fix:>13,}{f_fix/1e9:>14.4f}{f_spa/1e9:>15.4f}'
          f'{saving:>12.1f}%{theory:>8.1f}%')
    rows.append({'variant': name, 'channels': ch,
                 'params_fixed': p_fix, 'params_sparse': p_spa,
                 'gflops_fixed': f_fix / 1e9, 'gflops_sparse': f_spa / 1e9,
                 'flop_saving_pct': round(saving, 2),
                 'router_param_overhead_pct': round(100 * (p_spa - p_fix) / p_fix, 2)})

print(f'\n{"variant":<12}{"router param overhead":>24}')
print('-' * 36)
for r in rows:
    print(f'{r["variant"]:<12}{r["router_param_overhead_pct"]:>23.1f}%')

best, worst = rows[0], rows[-1]
print(f'\nFLOP saving falls from {best["flop_saving_pct"]:.1f}% at {best["variant"]} '
      f'to {worst["flop_saving_pct"]:.1f}% at {worst["variant"]}.')
print('Router parameter overhead moves the opposite way, so the method pays for '
      'itself only\nin the narrow-channel regime. Wide variants are the control that '
      'demonstrates this,\nnot arms where a gain should be expected.')

out = os.path.join(RESULTS, 'variant_scaling.json')
json.dump({'input': '1x3x256x256', 'rows': rows}, open(out, 'w'), indent=2)
print(f'\nwrote {out}')
