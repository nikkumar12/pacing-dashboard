@echo off
echo ====================================
echo  PaceSmart Full Refresh Started
echo ====================================

cd /d C:\Users\nikkumar12\PyCharmMiscProject

set PYTHON_PATH=C:\Users\nikkumar12\PyCharmMiscProject\.venv\Scripts\python.exe

echo.
echo Step 1: Generating variation raw data...
"%PYTHON_PATH%" generate_variation_raw_data.py

echo.
echo Step 2: Running Excel Pacing Generator...
"%PYTHON_PATH%" "Excel Pacing Generator.py"

echo.
echo Step 3: Running ML Script...
"%PYTHON_PATH%" "SINGLE PYTHON SCRIPT.py"

echo.
echo Step 4: Pushing to GitHub...
git add .
git commit -m "Auto refresh - %date% %time%"
git push origin main

echo.
echo DONE. Dashboard will refresh in 1-2 minutes.
pause
