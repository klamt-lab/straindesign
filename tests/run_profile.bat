@echo off
call C:\Users\phili\miniconda3\Scripts\activate.bat straindesign
cd /d C:\Users\phili\OneDrive\Dokumente\Python\straindesign
python tests/profile_preprocessing.py > tests/profile_out.txt 2>&1
echo Done. Exit code: %errorlevel%
