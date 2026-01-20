
$MaxRetries = 1000
$RetryCount = 0

Write-Host "🚀 Starting Robust Transcript Collection Loop..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow

while ($RetryCount -lt $MaxRetries) {
    $Date = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$Date] Starting pipeline..." -ForegroundColor Cyan

    # Run the python script
    # We use 'cmd /c' to avoid PowerShell capturing stderr aggressively if it crashes hard
    cmd /c "uv run python run_pipeline.py transcripts --input data/processed/videos_bac_only.csv --output data/processed/transcripts.csv --proxy socks5h://127.0.0.1:9150"
    
    $ExitCode = $LASTEXITCODE
    
    if ($ExitCode -eq 0) {
        Write-Host "✅ Collection completed successfully!" -ForegroundColor Green
        break
    }
    
    $RetryCount++
    Write-Host "⚠️ Script crashed or stopped (Exit Code: $ExitCode). Restarting in 60 seconds... (Attempt $RetryCount/$MaxRetries)" -ForegroundColor Red
    Start-Sleep -Seconds 60
}
