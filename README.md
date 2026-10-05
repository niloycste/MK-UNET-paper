# MK-UNet

Official Pytorch implementation of the paper [MK-UNet: Multi-kernel Lightweight CNN for Medical Image Segmentation](https://openaccess.thecvf.com/content/ICCV2025W/CVAMD/papers/Rahman_MK-UNet_Multi-kernel_Lightweight_CNN_for_Medical_Image_Segmentation_ICCVW_2025_paper.pdf) published in ICCV 2025 CVAMD
[Md Mostafijur Rahman](https://mostafij-rahman.github.io/), [Radu Marculescu](https://radum.ece.utexas.edu/)
<p>The University of Texas at Austin</p>

[ARXIV](https://arxiv.org/abs/2509.18493) | [PAPER](https://openaccess.thecvf.com/content/ICCV2025W/CVAMD/papers/Rahman_MK-UNet_Multi-kernel_Lightweight_CNN_for_Medical_Image_Segmentation_ICCVW_2025_paper.pdf) | [Code](https://github.com/SLDGroup/MK-UNet)

#### 🔍 **Check out our papers: [LoMix](https://github.com/SLDGroup/LoMix) [NeurIPS 2025], [EfficientMedNeXt](https://github.com/SLDGroup/EfficientMedNeXt) [MICCAI 2025], [EffiDec3D](https://github.com/SLDGroup/EffiDec3D) [CVPR 2025], [EMCAD](https://github.com/SLDGroup/EMCAD) [CVPR 2024], [PP-SAM](https://github.com/SLDGroup/PP-SAM) [CVPRW 2024], [G-CASCADE](https://github.com/SLDGroup/G-CASCADE) [WACV 2024], [MERIT](https://github.com/SLDGroup/MERIT) [MIDL 2023], [CASCADE](https://github.com/SLDGroup/CASCADE) [WACV 2023]**

## Local Reproduction and Path A Extension

This repository also contains a local two-step evaluation and extension study by **M. Mohaiminul Islam**. This added work is separate from the original MK-UNet authors' official release. The goal is to reproduce the released polyp pipeline, audit the code, and test a methodological extension called **MK-UNet-CC: Conditional and Sparse Multi-Kernel Computation**.

### Step 1: MK-UNet-T reproduction

The reproduction used the released polyp pipeline with `MK_UNet_T` on ClinicDB and ColonDB. Both runs used one seed, 200 epochs, repository-default hyperparameters, and CPU-only execution.

| Dataset | Paper DICE | Our DICE | Our IoU | Interpretation |
|---|---:|---:|---:|---|
| ClinicDB | 91.26 | 91.09 | 84.63 | Close to the published mean |
| ColonDB | 85.03 | 78.32 | 69.46 | Lower than the published mean |

The ClinicDB result is numerically consistent with the paper. The ColonDB result is not an exact reproduction of the paper result. The difference may come from the repository-default protocol, augmentation, preprocessing, random seed, or the lack of paper-exact multi-seed runs.

The archived run logs and evaluation spreadsheets are in `step1/logs/` and
`step1/results_polyp/`; the per-epoch curves are in `step1/deliverable2/figures/`.
The report source files are in `Pdf files/`. Those report PDFs are not included in
this checkout. To reproduce the experiments, use the commands below; they write
new outputs to the root-level `logs/` and `model_pth/` folders.

### Step 2 / Path A: MK-UNet-CC extension

The original MK-UNet multi-kernel depth-wise convolution block computes the `1x1`, `3x3`, and `5x5` branches for every input and combines them with a fixed sum:

```text
F = F1 + F3 + F5
```

The extension asks whether every image and every stage needs all three kernel branches equally. MK-UNet-CC keeps the MK-UNet topology but changes the multi-kernel computation in controlled stages:

1. **Adaptive soft routing:** a small router predicts input-dependent weights for the three branches. This tests adaptivity, but all branches still run.
2. **Predictive-entropy routing:** decoder routers can also receive a difficulty signal from earlier coarse segmentation heads. This uses predictive entropy, not Bayesian epistemic uncertainty.
3. **Sparse top-1 routing:** at inference, only the selected branch is executed. This is an exploratory efficiency experiment; training still evaluates all branches.
4. **Routing-guided branch-removal ablations:** routing statistics motivate static variants, but the tested branch subsets are exploratory and are not claimed to be an optimal pruning rule.

#### MK-UNet-CC architecture (our proposed extension)

<p align="center">
<img src="path%20A/results/mkunet_cc_architecture.png" width=100% class="center">
</p>

This figure shows the proposed extension on top of the original MK-UNet topology. The CMKDC-CC block adds routing to the multi-kernel depth-wise branches, the decoder can use detached predictive-entropy signals from coarse heads, and post-training routing statistics can be used for static pruning.

### Preliminary extension results

All extension results below are preliminary, single-seed ClinicDB results. Runs with different epoch budgets are not treated as directly comparable.

| Arm | Epochs | Params | Test DICE | FLOPs at 256 | Batch-1 CPU latency | Main finding |
|---|---:|---:|---:|---:|---:|---|
| Fixed MK-UNet-T baseline | 200 | 27,384 | 91.09 | 0.0668 G | 53.6 ms | Reproduction baseline |
| Adaptive soft routing | 200 | 30,588 | 91.53 | 0.0677 G | 57.5 ms | Small accuracy gain in one run, but not sparse |
| Fixed + auxiliary heads | 100 | 27,384 | 88.89 | 0.0668 G | - | Matched control for entropy experiment |
| Adaptive + auxiliary heads | 100 | 30,588 | 89.97 | 0.0677 G | - | Adaptivity helped over matched control |
| Predictive entropy + auxiliary heads | 100 | 30,618 | 88.99 | 0.0677 G | - | Entropy did not improve DICE in this run |
| Sparse top-1 | 60 fine-tune | 30,588 | 90.01 | 0.0307 G | 46.4 ms | Real branch skipping and lower latency |

In the preliminary batch-1 ClinicDB measurement, forward hooks verified that only **10 of 30** branch convolutions executed and **20 of 30** were skipped. The tool-reported FLOPs were about **54.1%** lower and the measured CPU latency about **13.3%** lower for that setup. These are local measurements, not guarantees for other batch sizes, devices, or implementations; routing and dynamic dispatch add runtime overhead.

The current result should not be described as final proof that MK-UNet-CC is better than MK-UNet. The safe claim is narrower: **conditional sparse branch execution can reduce computation and measured CPU latency while keeping competitive ClinicDB performance in this preliminary single-seed study**. Final claims require full-budget, multi-seed runs and target-device latency tests.

Key extension artifacts (results are a snapshot and may be refreshed as the sweep progresses):

- `path A/results/all_runs.csv` — collected run table
- `path A/results/cost_table.json` — parameter and FLOP comparison
- `path A/results/sparse_verification.json` — executed/skipped branch counts and latency
- `path A/results/mkunet_cc_architecture.png` — extension architecture figure
- `submission_texfile_pdf/MKUNet_All_Revised_v4/Step2_PathA_Research_Proposal.pdf` — final Path A proposal with the method framing, literature positioning, and limitations

## Reproducing the experiments

### Environment

The dependency file is based on the paper's Python 3.8 / PyTorch 1.11 stack. Create and activate an environment, then install `requirements.txt`:

```powershell
conda create -n mkunetenv python=3.8 -y
conda activate mkunetenv
pip install -r requirements.txt
```

The default requirements select CUDA 11.3 PyTorch wheels; they can also run on a CPU-only machine, although they use more memory. For a CPU-only install, follow the CPU wheel alternatives documented at the top of `requirements.txt`. The resolved environment used for the archived reproduction is recorded in `step1/environment/pip_freeze.txt`.

### Data

Datasets are not committed. Download them from their original sources and prepare the following split structure (images and masks must have matching filenames):

```text
data/polyp/target/ClinicDB/{train,val,test}/{images,masks}/
data/polyp/target/ColonDB/{train,val,test}/{images,masks}/
```

The archived split sizes are ClinicDB 489/61/62 and ColonDB 303/38/38 for train/validation/test. Run all commands below from the repository root. Check the prepared data with:

```powershell
python -W ignore step1/scripts/check_data.py
```

### Step 1: reproduce the MK-UNet-T baseline

The reported baseline used one run per dataset, seed 42, 200 epochs, CPU, and the training script's default image size, batch size, learning rate, and augmentation. Train each dataset:

```powershell
python -W ignore train_polyp.py --network MK_UNet_T --dataset ClinicDB --epoch 200 --runs 1 --seed 42 --device cpu
python -W ignore train_polyp.py --network MK_UNet_T --dataset ColonDB --epoch 200 --runs 1 --seed 42 --device cpu
```

Record the `run_id` printed by each training run, then evaluate its test split:

```powershell
python -W ignore test_polyp.py --run_id <run_id> --network MK_UNet_T --dataset_name ClinicDB --split test --img_size 352 --device cpu
python -W ignore test_polyp.py --run_id <run_id> --network MK_UNet_T --dataset_name ColonDB --split test --img_size 352 --device cpu
```

Fresh training logs and checkpoints are saved under root-level `logs/` and `model_pth/`. These outputs are ignored by Git. The archived results in `step1/` are provided for reference. The reported scores are single-seed observations, not a multi-seed or exact reproduction claim.

### Step 2: run the Path A ClinicDB experiments

Prepare ClinicDB in the structure above and install the requirements. Run the checks from the repository root:

```powershell
python -W ignore "path A/01_verify_head_mapping.py"
python -W ignore "path A/02_smoke_test.py"
python -W ignore "path A/03_measure_cost.py"
python -W ignore "path A/04_verify_sparse_execution.py"
```

Then run the staged ClinicDB queue (adaptive baseline, controls, and sparse fine-tuning):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "path A/run_queue.ps1" -Python python -Device cpu
```

Use `-Device cuda` for a CUDA run. This queue is the ClinicDB ablation workflow, not the broader multi-dataset sweep. It does not resume an interrupted training run automatically. Refresh consolidated tables and figures with:

```powershell
python -W ignore "path A/11_finalize.py"
```

Run outputs are written to `path A/runs/`; summary metrics and analysis are in `path A/results/`.

### Optional: package the broader multi-machine sweep

The separate shard workflow generates machine-specific scripts and packages under `dist/`. Generate three shards and build the packages with:

```powershell
python -W ignore "path A/13_make_shards.py" --shards 3
python -W ignore "path A/14_package_machines.py"
```

See `HANDOVER.md` for how to run and collect those packages. They are generated deployment copies, not required for reproducing the Step 1 baseline or the Path A ClinicDB queue.

## Architecture

<p align="center">
<img src="mkunet_architecture.png" width=100% height=40% 
class="center">
</p>

## Quantitative Results

## Qualitative Results

## Usage:
### Recommended environment:
**Please run the following commands.**
```
conda create -n mkunetenv python=3.8
conda activate mkunetenv

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

pip install -r requirements.txt

```

### Data preparation:

- **ClinicDB dataset:**
Download the splited ClinicDB dataset from [Google Drive](https://drive.google.com/drive/folders/1FPJr5f91uUCikxMvkwtZSEnYHemTZq1P?usp=share_link) and move into './data/polyp/' folder.

- **ColonDB dataset:**
Download the splited ColonDB dataset from [Google Drive](https://drive.google.com/drive/folders/1u4_8dMztnEBUaX-w3XfUR3jXLBhpccPA?usp=share_link) and move into './data/polyp/' folder.

### Training:
```
cd into MK-UNet
CUDA_VISIBLE_DEVICES=0 python -W ignore train_polyp.py --network MK_UNet

```

### Testing:
```
cd into MK-UNet 
CUDA_VISIBLE_DEVICES=0 python -W ignore test_polyp.py --network MK_UNet --run_id <your run_id>

```

## Acknowledgement
We are very grateful for these excellent works [EMCAD](https://github.com/SLDGroup/EMCAD), [CASCADE](https://github.com/SLDGroup/CASCADE), [MERIT](https://github.com/SLDGroup/MERIT), [G-CASCADE](https://github.com/SLDGroup/G-CASCADE), [PP-SAM](https://github.com/SLDGroup/PP-SAM), [PraNet](https://github.com/DengPingFan/PraNet), and [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT), which have provided the basis for our framework.

## Citations

``` 
@inproceedings{rahman2025mk,
  title={Mk-unet: Multi-kernel lightweight cnn for medical image segmentation},
  author={Rahman, Md Mostafijur and Marculescu, Radu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1042--1051},
  year={2025}
}
```
