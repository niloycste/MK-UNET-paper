@echo off
cd /d C:\MK-UNet
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\MK-UNet\path A\run_new_datasets.ps1" > "C:\MK-UNet\path A\datasets.log" 2>&1
