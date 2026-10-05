# Restarts machineA's sweep if it has stopped, and does nothing if it is running.
#
# The task's logon trigger only covers a reboot. This covers the other case: the
# orchestrator dies (crash, accidental close, killed process) while the machine stays
# up. Without it the machine would sit idle until someone noticed -- on a 39-day run
# that could be days of lost time.
#
# The guard matters more than the restart: starting a second sweep alongside a live
# one would run two training jobs at once, which has already caused an out-of-memory
# failure in this project.

$run = 'C:\MK-UNet\dist\machineA\run.ps1'
$log = 'C:\MK-UNet\dist\machineA\machineA.log'

# A scheduled task gets no conda activation, so bare "python" resolves to whatever is
# on the system PATH -- an interpreter without numpy. run.ps1 honours MKUNET_PY, so
# set it explicitly. Omitting this once made the sweep burn through every unit in
# seconds with ModuleNotFoundError instead of training.
$env:MKUNET_PY = 'C:\Users\USER\.conda\envs\mkunetenv\python.exe'

function Note($m) {
    "$(Get-Date -Format 'MM-dd HH:mm:ss')  [watchdog] $m" | Add-Content -Path $log -Encoding utf8
}

$training = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
              Where-Object { $_.CommandLine -like '*05_train.py*' })
if ($training.Count -gt 0) { exit 0 }        # healthy: say nothing, keep the log clean

# Match only a real "-File ...\run.ps1" launch. A plain -like on the path also matches
# any shell whose command text merely mentions it -- including a monitoring command --
# and a false positive here silently suppresses the restart, which is the one thing
# this script exists to do.
$orch = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
          Where-Object { $_.CommandLine -match '-File\s+"?[^"]*machineA\\run\.ps1' })
if ($orch.Count -gt 0) { exit 0 }            # between arms, about to launch the next

Note 'no training and no orchestrator found -- restarting the sweep'

# Refuse to start on a broken interpreter. Without this the sweep "runs" to completion
# in seconds, every unit failing on import, and the watchdog then restarts it every
# 2 hours forever -- looking busy while doing nothing.
& $env:MKUNET_PY -c "import numpy, torch" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Note "ABORT: $env:MKUNET_PY cannot import numpy/torch -- fix the environment"
    exit 1
}

# Launched with -File so the child's command line contains "machineA\run.ps1",
# which is exactly what the $orch check above looks for on the next tick.
& powershell -NoProfile -ExecutionPolicy Bypass -File $run *>&1 |
    Add-Content -Path $log -Encoding utf8

Note "sweep exited (code $LASTEXITCODE)"
