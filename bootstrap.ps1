$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m owywvad deps install

Write-Host "Environment bootstrapped. Run 'python -m owywvad --help' from githubcode/."

