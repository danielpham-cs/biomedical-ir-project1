# Run Helix as a server reachable by devices on the NCKU local network.
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

& .\.venv\python.exe -m streamlit run app.py `
    --server.address 0.0.0.0 `
    --server.port 8501 `
    --server.headless true
