"""Convert every raw benchmark into the layout the MK-UNet pipeline expects.

Output follows the released repository's own convention, so nothing downstream
changes:

    data/<task>/target/<Dataset>/{train,val,test}/{images,masks}/*.png

exactly as data/polyp/target/ClinicDB already ships. Task folders mirror the
paper's four application groups:

    data/busi/target/BUSI          breast cancer     647 images @ 256
    data/isic/target/ISIC18        skin lesion     2,594 images @ 256
    data/polyp/target/ClinicDB     polyp             612 images @ 352  (already present)
    data/polyp/target/ColonDB      polyp             379 images @ 352  (already present)
    data/cell/target/DSB18         cell nuclei       670 images @ 256
    data/cell/target/EM            cell structure     30 images @ 256

Train with, e.g., --data_root data/cell/target --dataset DSB18.

Datasets are handled independently, so they can be prepared as the data arrives:

    python -W ignore prepare_datasets.py                # every dataset present
    python -W ignore prepare_datasets.py --dataset busi # just one
    python -W ignore prepare_datasets.py --dataset isic18   # later, once downloaded

Splits use a FIXED seed, so preparing a dataset today and another next week still
gives reproducible, disjoint 80:10:10 partitions. Re-running a dataset reproduces
exactly the same split.

Per-dataset handling (each is a real quirk of the source data, not boilerplate):

  BUSI     `normal/` is EXCLUDED. The paper reports 647 images = 437 benign + 210
           malignant; including the 133 normal cases would give 780 and would add
           images with no lesion to a lesion-segmentation benchmark.
           18 cases carry several mask files (`_mask.png`, `_mask_1.png`, ...) for
           multiple lesions; these are merged with a logical OR. Without this the
           mask count (665) exceeds the image count (647) and pairing silently
           shifts.

  DSB18    Source masks are one PNG per nucleus. `stage1_train_combinedmasks/`
           already holds the merged binary masks, so those are used directly.
           `stage1_test/` is ignored: its labels were never released publicly.

  EM       ISBI 2012 ships 30 slices inside ONE multi-page TIFF, so the stack is
           split into individual frames. ISBI polarity is KEPT: white = cell
           interior = foreground (fraction 0.780).

           An earlier version inverted this to make membrane the foreground, on
           the reasoning that membrane is the structure of interest. That scored
           71.39 Dice against the paper's 94.69. The polarity is what decides it:
           a trivial all-foreground predictor scores 0.877 on the interior task
           but only 0.360 on the membrane task, so 94.69 is only reachable in the
           un-inverted orientation. Use --em_invert to reproduce the old (wrong)
           behaviour. Only 30 labelled images exist, so the split is 24/3/3.

  ISIC18   Masks carry a suffix the images do not: ISIC_0000000.jpg pairs with
           ISIC_0000000_segmentation.png, so pairing is by stem, never by sort
           order.
"""
import argparse
import os
import random
import shutil
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, 'data')

SPLIT_SEED = 42          # fixed so datasets prepared at different times stay reproducible
SPLIT = (0.8, 0.1, 0.1)

# Output follows the released repository's own convention,
#     data/<task>/target/<Dataset>/{train,val,test}/{images,masks}/
# which is how data/polyp/target/ClinicDB already ships. Task folders mirror the
# paper's four application groups (breast cancer, skin lesion, polyp, cell), so
# the existing loader works unchanged via --data_root data/<task>/target.
TASK = {'busi': 'busi', 'isic18': 'isic', 'dsb18': 'cell', 'em': 'cell'}
DSNAME = {'busi': 'BUSI', 'isic18': 'ISIC18', 'dsb18': 'DSB18', 'em': 'EM'}


def _dest(key):
    return os.path.join(RAW, TASK[key], 'target', DSNAME[key])


# --------------------------------------------------------------------------- utils
def _resize(img, size, is_mask):
    # nearest for masks so labels stay binary; bilinear for images
    return img.resize((size, size), Image.NEAREST if is_mask else Image.BILINEAR)


def _open(x):
    """Accept either an already-loaded PIL image or a path to open on demand.

    Builders that synthesise a mask (BUSI OR-merges multi-lesion masks) must pass the
    image itself; builders that just read files should pass paths. Holding 2594
    full-resolution ISIC18 images open at once exhausts memory, so that one streams.
    """
    return Image.open(x) if isinstance(x, str) else x


def _write_split(pairs, key, size, binarize_at=127):
    """pairs: list of (stem, image, mask), each of which may be a PIL image or a path.

    Writes a deterministic 80:10:10.
    """
    pairs = sorted(pairs, key=lambda p: p[0])          # stable order before shuffling
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(pairs)

    n = len(pairs)
    n_tr = int(round(n * SPLIT[0]))
    n_va = int(round(n * SPLIT[1]))
    # give any rounding remainder to test, and guarantee val/test are non-empty
    n_va = max(1, n_va)
    n_te = max(1, n - n_tr - n_va)
    n_tr = n - n_va - n_te

    parts = {'train': pairs[:n_tr],
             'val':   pairs[n_tr:n_tr + n_va],
             'test':  pairs[n_tr + n_va:]}

    base = _dest(key)
    if os.path.isdir(base):
        shutil.rmtree(base)                            # re-running must not merge old output
    for split, items in parts.items():
        for sub in ('images', 'masks'):
            os.makedirs(os.path.join(base, split, sub), exist_ok=True)
        for stem, img_src, msk_src in items:
            im, mk = _open(img_src), _open(msk_src)
            try:
                img = _resize(im.convert('RGB'), size, False)
                msk = _resize(mk.convert('L'), size, True)
            finally:
                # Close only what was opened here; a caller-supplied image may be reused.
                if isinstance(img_src, str):
                    im.close()
                if isinstance(msk_src, str):
                    mk.close()
            m = (np.array(msk) > binarize_at).astype(np.uint8) * 255
            img.save(os.path.join(base, split, 'images', f'{stem}.png'))
            Image.fromarray(m).save(os.path.join(base, split, 'masks', f'{stem}.png'))

    print(f'  {DSNAME[key]}: {n} images -> train {len(parts["train"])} / '
          f'val {len(parts["val"])} / test {len(parts["test"])}  @ {size}x{size}')
    return n


# --------------------------------------------------------------------------- BUSI
def prepare_busi(size=256):
    src = os.path.join(RAW, 'Dataset_BUSI_with_GT')
    if not os.path.isdir(src):
        return None
    pairs, merged = [], 0
    for cls in ('benign', 'malignant'):                # 'normal' deliberately excluded
        d = os.path.join(src, cls)
        if not os.path.isdir(d):
            continue
        files = os.listdir(d)
        images = sorted(f for f in files if '_mask' not in f and f.lower().endswith('.png'))
        for fn in images:
            stem = os.path.splitext(fn)[0]
            masks = sorted(f for f in files
                           if f.startswith(stem + '_mask') and f.lower().endswith('.png'))
            if not masks:
                print(f'    WARNING: no mask for {cls}/{fn}, skipped')
                continue
            acc = None
            for mf in masks:                            # logical OR across multiple lesions
                a = np.array(Image.open(os.path.join(d, mf)).convert('L'))
                acc = a if acc is None else np.maximum(acc, a)
            if len(masks) > 1:
                merged += 1
            pairs.append((f'{cls}_{stem}'.replace(' ', '_'),
                          Image.open(os.path.join(d, fn)),
                          Image.fromarray(acc)))
    print(f'  BUSI: merged multi-lesion masks in {merged} cases; '
          f"'normal' class excluded")
    return _write_split(pairs, 'busi', size)


# --------------------------------------------------------------------------- DSB18
def prepare_dsb18(size=256):
    imgs = os.path.join(RAW, 'data-science-bowl-2018', 'stage1_train')
    msks = os.path.join(RAW, 'data-science-bowl-2018', 'stage1_train_combinedmasks')
    if not (os.path.isdir(imgs) and os.path.isdir(msks)):
        return None
    pairs = []
    for entry in sorted(os.listdir(imgs)):
        ip = os.path.join(imgs, entry, 'images', f'{entry}.png')
        mp = os.path.join(msks, f'{entry}.png')
        if not (os.path.isfile(ip) and os.path.isfile(mp)):
            continue
        pairs.append((entry[:16], Image.open(ip), Image.open(mp)))
    return _write_split(pairs, 'dsb18', size)


# --------------------------------------------------------------------------- EM
def prepare_em(size=256, invert=False):
    base = os.path.join(RAW, 'electron microscopy data')
    vol = os.path.join(base, 'train-volume.tif')
    lab = os.path.join(base, 'train-labels.tif')
    if not (os.path.isfile(vol) and os.path.isfile(lab)):
        return None
    v, l = Image.open(vol), Image.open(lab)
    n = min(getattr(v, 'n_frames', 1), getattr(l, 'n_frames', 1))
    pairs = []
    for i in range(n):
        v.seek(i); l.seek(i)
        m = np.array(l.convert('L'))
        if invert:
            m = 255 - m                                 # membrane becomes foreground
        pairs.append((f'em_{i:03d}', v.convert('RGB').copy(), Image.fromarray(m)))
    fg = np.mean([(np.array(p[2]) > 127).mean() for p in pairs])
    print(f'  EM: {n} slices from the stack; labels '
          f'{"INVERTED (membrane = foreground)" if invert else "left as-is"}; '
          f'foreground fraction {fg:.3f}')
    return _write_split(pairs, 'em', size)


# --------------------------------------------------------------------------- ISIC18
def prepare_isic18(size=256):
    """Run this once ISIC2018_Task1-2_Training_Input + _GroundTruth are extracted."""
    base = os.path.join(RAW, 'isic18')
    img_dir = msk_dir = None
    for dp, _, fs in os.walk(base):
        low = os.path.basename(dp).lower()
        if any(f.lower().endswith(('.jpg', '.jpeg')) for f in fs) and 'input' in low:
            img_dir = dp
        if any(f.lower().endswith('.png') for f in fs) and 'groundtruth' in low:
            msk_dir = dp
    if not (img_dir and msk_dir):                      # not downloaded yet
        return None
    masks = {os.path.splitext(f)[0].replace('_segmentation', ''): os.path.join(msk_dir, f)
             for f in os.listdir(msk_dir) if f.lower().endswith('.png')}
    pairs = []
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith(('.jpg', '.jpeg')):
            continue
        stem = os.path.splitext(f)[0]
        if stem not in masks:                          # pair by STEM, never by sort order
            print(f'    WARNING: no mask for {f}, skipped')
            continue
        # Paths, not open images: these are full-resolution dermoscopy files (up to
        # 6748x4499) and holding all 2594 open at once raises MemoryError.
        pairs.append((stem, os.path.join(img_dir, f), masks[stem]))
    return _write_split(pairs, 'isic18', size)


# --------------------------------------------------------------------------- main
BUILDERS = {'busi': prepare_busi, 'dsb18': prepare_dsb18,
            'em': prepare_em, 'isic18': prepare_isic18}
# The paper's reported sizes, used as a correctness check.
EXPECTED = {'busi': 647, 'dsb18': 670, 'em': 30, 'isic18': 2594}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=list(BUILDERS) + ['all'], default='all')
    ap.add_argument('--size', type=int, default=256,
                    help="output resolution; the paper uses 256 for these four "
                         "(polyp uses 352 and is already prepared)")
    ap.add_argument('--em_invert', action='store_true',
                    help='invert EM labels so membrane is foreground; this does NOT '
                         'reproduce the paper (see the EM note above)')
    a = ap.parse_args()

    todo = list(BUILDERS) if a.dataset == 'all' else [a.dataset]
    print('output -> data/<task>/target/<Dataset>/   (released-repo convention)\n')

    done, skipped = {}, []
    for name in todo:
        kw = {'size': a.size}
        if name == 'em':
            kw['invert'] = a.em_invert
        n = BUILDERS[name](**kw)
        if n is None:
            skipped.append(name)
            print(f'  {name}: raw data not found, skipped')
        else:
            done[name] = n

    print('\n--- count check against the paper ---')
    for name, n in done.items():
        exp = EXPECTED[name]
        print(f'  {name:8s} {n:5d}   paper {exp:5d}   '
              f'{"MATCH" if n == exp else f"MISMATCH ({n-exp:+d})"}')
    for name in skipped:
        print(f'  {name:8s}     -   paper {EXPECTED[name]:5d}   not prepared yet')

