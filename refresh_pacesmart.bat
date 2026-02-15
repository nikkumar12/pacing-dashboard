@echo off
echo ======================================
echo   PaceSmart Auto Refresh Started
echo ======================================

cd C:\Users\nikkumar12\PyCharmMiscProject

echo.
echo Step 1: Generating variation raw data...
C:\Users\nikkumar12\PyCharmMiscProject\.venv\Scripts\python.exe generate_variation_raw_data.py

echo.
echo Step 2: Running ML + Excel pipeline...
C:\Users\nikkumar12\PyCharmMiscProject\.venv\Scripts\python.exe "SINGLE PYTHON SCRIPT.py"

echo.
echo Step 3: Committing to Git...
git add .
git commit -m "Auto refresh pacing dashboard"

echo.
echo Step 4: Pushing to GitHub...
git push origin main

echo.
echo ======================================
echo   Dashboard Will Refresh Automatically
echo ======================================

pause