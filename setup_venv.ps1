# Setup script for virtual environment
# Run this script on each device after cloning the repository

Write-Host "Setting up virtual environment..." -ForegroundColor Cyan

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python not found. Please install Python first." -ForegroundColor Red
    exit 1
}

# Remove old venv if it exists
if (Test-Path "ypd_venv") {
    Write-Host "Removing old virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "ypd_venv"
}

# Create new virtual environment
Write-Host "Creating new virtual environment..." -ForegroundColor Cyan
python -m venv ypd_venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& ".\ypd_venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install requirements
if (Test-Path "requirements.txt") {
    Write-Host "Installing requirements from requirements.txt..." -ForegroundColor Cyan
    pip install -r requirements.txt
} else {
    Write-Host "Warning: requirements.txt not found!" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Setup complete! Virtual environment is ready." -ForegroundColor Green
Write-Host "To activate in the future, run: .\ypd_venv\Scripts\Activate.ps1" -ForegroundColor Cyan
