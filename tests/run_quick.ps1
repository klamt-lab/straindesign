Set-Location 'C:\Users\phili\OneDrive\Dokumente\Python\straindesign'
$py = 'C:\Users\phili\miniconda3\envs\straindesign\python.exe'
& $py tests/test_quick.py
Write-Host "Exit: $LASTEXITCODE"
