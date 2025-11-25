# Virtual Environment Quick Guide

## How to Check Which Venv is Active

**Method 1: Look at your PowerShell prompt**
- If you see `(venv_name)` at the start, that venv is active
- Example: `(ypd_venv) PS C:\...` means `ypd_venv` is active

**Method 2: Check Python path**
```powershell
Get-Command python | Select-Object -ExpandProperty Source
```
- Shows the full path to the Python executable
- If it contains `ypd_venv\Scripts\python.exe`, then `ypd_venv` is active
- If it contains `Pitch\Scripts\python.exe`, then `Pitch` is active

## How to Activate a Virtual Environment

**In PowerShell:**
```powershell
.\ypd_venv\Scripts\Activate.ps1
```

**If you get an execution policy error:**
```powershell
powershell -ExecutionPolicy Bypass -File .\ypd_venv\Scripts\Activate.ps1
```

**Or set it once (recommended):**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## How to Deactivate

**In PowerShell:**
```powershell
deactivate
```

**Or just close the terminal** - venvs only last for that session

## How to Check if a Package is Installed

```powershell
python -c "import pandas; print('pandas is installed')"
```

**Or list all installed packages:**
```powershell
pip list
```

## How to Install Missing Packages

```powershell
pip install pandas
```

**Or install from requirements.txt:**
```powershell
pip install -r requirements.txt
```

## Common Issues

**Problem:** `ModuleNotFoundError: No module named 'pandas'`
- **Solution:** You're in the wrong venv OR the package isn't installed
  1. Check which venv is active (see above)
  2. Activate the correct venv (`ypd_venv` for this project)
  3. If still missing, install: `pip install pandas`

**Problem:** Can't activate venv (execution policy error)
- **Solution:** Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## For This Project

**Always use `ypd_venv` for Action Plus scripts:**
```powershell
.\ypd_venv\Scripts\Activate.ps1
python "Action Plus\main.py"
```

