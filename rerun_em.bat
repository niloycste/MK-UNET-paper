@echo off
REM Lives at the repo root, NOT in "path A", because schtasks stores /tr unquoted:
REM a space in the action path makes it resolve "C:\MK-UNet\path" and fail with
REM 0x80070002. The working MKUNetDatasets task uses the same wrapper pattern.
REM
REM No `>>` redirect here on purpose -- rerun_em.ps1 appends to its own log per
REM write, so a stale instance can never hold the file locked.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\MK-UNet\path A\rerun_em.ps1"
