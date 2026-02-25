Set-Location 'C:\Users\phili\OneDrive\Dokumente\Python\straindesign'
$py = 'C:\Users\phili\miniconda3\envs\straindesign\python.exe'
& $py tests/profile_preprocessing.py 2>&1 | Out-File -FilePath tests/profile_out.txt -Encoding utf8
Write-Host "Exit: $LASTEXITCODE"
