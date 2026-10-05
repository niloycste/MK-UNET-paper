# Re-run the EM validation baseline after the label-polarity fix.
#
# The first attempt scored 71.39 Dice against the paper's 94.69 because
# prepare_datasets.py inverted the ISBI labels, making thin membrane the
# foreground (fraction 0.220) instead of cell interior (0.780). A trivial
# all-foreground predictor scores 0.360 on the inverted task and 0.877 on the
# correct one, so 94.69 is only reachable in the un-inverted orientation.
#
# The data has been re-prepared with the same SPLIT_SEED, so the image split is
# byte-identical to the first attempt and polarity is the only variable.

$py  = "C:\Users\USER\.conda\envs\mkunetenv\python.exe"
$log = "C:\MK-UNet\path A\em_rerun.log"
Set-Location "C:\MK-UNet"

# Log by appending per write rather than through a shell `>>` redirect. A redirect
# holds the handle for the life of the process, so a stale instance kept the file
# locked and every later run died with exit 1 before producing any output.
function Log($m) {
    "$(Get-Date -Format 'MM-dd HH:mm:ss')  $m" | Add-Content -Path $log -Encoding utf8
}

# Never run two training jobs at once (~7 GB RSS each caused an OOM earlier).
#
# This waits on the MKUNetDatasets TASK, not on python process ages. An age-based
# guard polling every 300 s and treating a process younger than 10 min as idle is
# guaranteed to poll inside the 600 s window where BUSI has ended and DSB18 has
# just begun -- it would break early and run EM alongside DSB18. The task state has
# no such gap: it stays Running across the whole queue.
Log "waiting for the MKUNetDatasets queue to drain"
while (schtasks /query /tn MKUNetDatasets /fo list 2>$null | Select-String 'Status:\s+Running') {
    Start-Sleep -Seconds 300
}

Log "START EM re-run, corrected polarity -- paper 94.69, previous (inverted) 71.39"
& $py -W ignore "path A\05_train.py" `
    --dataset EM --data_root data/cell/target --img_size 256 `
    --routing_mode fixed --kernel_sizes 1 3 5 `
    --epoch 100 --lr 0.0005 --aux_supervision False `
    --runs 1 --seed 42 --device cpu 2>&1 | Add-Content -Path $log -Encoding utf8

Log "DONE EM re-run (exit $LASTEXITCODE)"
