# Project Setup Guide

## First Time Setup (on each device)

1. Clone the repository
2. Run the setup script:
   ```powershell
   .\setup_venv.ps1
   ```

## Daily Usage

1. Activate the virtual environment:
   ```powershell
   .\ypd_venv\Scripts\Activate.ps1
   ```

2. Run your scripts:
   ```powershell
   python "Action Plus\main.py"
   python "Curveball Test\main.py"
   ```

3. When done, deactivate (optional):
   ```powershell
   deactivate
   ```

## Adding New Packages

If you install a new package, update requirements.txt:
```powershell
pip freeze > requirements.txt
```

Then commit the updated `requirements.txt` so other devices can install it.
```

**4. Remove the venv from git (if it's already tracked):**

If `ypd_venv` is already tracked by git, remove it:

```powershell
# Remove from git tracking (but keep local files)
git rm -r --cached ypd_venv/

# Commit the removal
git commit -m "Remove venv from git tracking"
```

**5. Update `requirements.txt` to include all dependencies:**

Check what's installed and update requirements.txt:

```powershell
# After activating venv
pip freeze > requirements.txt
```

**Summary:**
- Update `.gitignore` to exclude `ypd_venv/` and other machine-specific files
- Create `setup_venv.ps1` that each device runs to create a local venv
- Remove the venv from git if it's tracked
- Keep `requirements.txt` up to date

Each device will have its own venv with paths matching that machine, avoiding path conflicts. Run `setup_venv.ps1` on each device after pulling changes.
