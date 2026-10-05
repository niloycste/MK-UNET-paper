# Baseline (fixed [1,3,5]) on each newly prepared dataset.
#
# PURPOSE: validate the data preparation, not to produce final results.
#
# prepare_datasets.py made choices that could silently corrupt a benchmark --
# excluding BUSI's 'normal' class, OR-merging 17 multi-lesion masks, using DSB18's
# pre-merged masks, splitting the EM stack and inverting its label polarity. If any
# of those is wrong, DICE will land far from the paper's figure. Catching that here
# costs a few CPU hours; catching it after a few hundred GPU-hours costs days.
#
# Reference (MK-UNet-T, paper Table 1: 200 epochs, 5-run mean, paper protocol):
#     BUSI 75.64    DSB18 92.38    EM 94.69
# These runs are 100 epochs, single seed, repository-default hyperparameters, so
# they should land in the same neighbourhood -- NOT match exactly. A result within
# a few points confirms the data is sound; a result 30+ points adrift means a
# preparation bug.
#
# EM is only 24 training images, so its run takes minutes and goes first as the
# fastest possible smoke signal.

$py = "C:\Users\USER\.conda\envs\mkunetenv\python.exe"
Set-Location "C:\MK-UNet"

# Never run two training jobs at once (~7 GB RSS each).
$waited = 0
while ($true) {
    $busy = Get-Process python -ErrorAction SilentlyContinue |
            Where-Object { $_.StartTime -lt (Get-Date).AddMinutes(-10) }
    if (-not $busy) { break }
    if ($waited -eq 0) { Write-Output "Waiting for in-flight training..." }
    Start-Sleep -Seconds 180
    $waited += 3
}

$jobs = @(
  @{ n = "1/4 EM      (24 train,  256px)  paper 94.69";
     a = @("--dataset","EM","--data_root","data/cell/target","--img_size","256") },
  @{ n = "2/4 BUSI    (518 train, 256px)  paper 75.64";
     a = @("--dataset","BUSI","--data_root","data/busi/target","--img_size","256") },
  @{ n = "3/4 DSB18   (536 train, 256px)  paper 92.38";
     a = @("--dataset","DSB18","--data_root","data/cell/target","--img_size","256") },
  @{ n = "4/4 ColonDB (303 train, 352px)  paper 85.03  [extension baseline @100ep]";
     a = @("--dataset","ColonDB","--data_root","data/polyp/target","--img_size","352") }
)

$common = @("--routing_mode","fixed","--kernel_sizes","1","3","5",
            "--epoch","100","--lr","0.0005","--aux_supervision","False",
            "--runs","1","--seed","42","--device","cpu")

foreach ($j in $jobs) {
    Write-Output "=========================================================="
    Write-Output "START $($j.n)   [$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')]"
    Write-Output "=========================================================="
    & $py -W ignore "path A\05_train.py" @($j.a) @common
    if ($LASTEXITCODE -ne 0) {
        Write-Output "STOPPED at $($j.n) (exit $LASTEXITCODE)."
        exit $LASTEXITCODE
    }
    Write-Output "DONE  $($j.n)   [$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')]"
}

Write-Output "All dataset-validation baselines complete."
