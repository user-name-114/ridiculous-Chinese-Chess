@echo off
if exist "C:\Users\Lenovo\anaconda3\pythonw.exe" (
    start "" "C:\Users\Lenovo\anaconda3\pythonw.exe" "%~dp0panel.py"
) else (
    where pythonw >nul 2>nul
    if %errorlevel%==0 (
        start "" pythonw "%~dp0panel.py"
    ) else (
        echo [Error] pythonw not found. Install Python and retry.
        pause
    )
)
