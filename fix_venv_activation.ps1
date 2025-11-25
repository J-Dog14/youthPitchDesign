# Script to fix virtual environment activation in PowerShell
# Run this once to set your execution policy

Write-Host "Setting PowerShell execution policy to allow local scripts..." -ForegroundColor Yellow
Write-Host "This allows you to activate your virtual environment normally." -ForegroundColor Yellow
Write-Host ""

# Set execution policy for current user (doesn't require admin)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

Write-Host "Execution policy updated successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "You can now activate your virtual environment using:" -ForegroundColor Cyan
Write-Host "  .\ypd_venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Or simply:" -ForegroundColor Cyan
Write-Host "  ypd_venv\Scripts\Activate.ps1" -ForegroundColor White

