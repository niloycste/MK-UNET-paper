@echo off
REM Entry point for the MKUNetMachineA scheduled task.
REM
REM Delegates to watchdog_machineA.ps1, which starts the sweep only if it is not
REM already running. That makes the task safe to fire repeatedly: the logon trigger
REM covers reboots and the repetition trigger covers an orchestrator that died while
REM the machine stayed up.
REM
REM Lives at the repo root because schtasks stores /tr unquoted and a space in the
REM action path fails with 0x80070002.
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\MK-UNet\watchdog_machineA.ps1"
