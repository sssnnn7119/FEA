@echo off
echo Starting FEA UI...
python launch_fea_ui.py
if errorlevel 1 (
    echo Error launching the FEA UI. Please check if Python is installed and in your PATH.
    pause
)
echo FEA UI closed.
pause