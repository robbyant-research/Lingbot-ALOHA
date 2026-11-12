$ErrorActionPreference = 'Stop'
$pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
if (-not $pythonCommand) { throw 'Python 3 is required to preview this site.' }
$serverPath = Join-Path $PSScriptRoot 'preview_server.py'
$probe = [System.Net.Sockets.TcpClient]::new()
try { $probe.Connect('127.0.0.1', 8000); $portBusy = $true } catch { $portBusy = $false } finally { $probe.Dispose() }
if ($portBusy) { throw 'Port 8000 is in use. Run preview_server.py with --port 8001 instead.' }
$previewProcess = Start-Process -FilePath $pythonCommand.Source -ArgumentList @('-u', ('"{0}"' -f $serverPath)) -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru
Set-Content -LiteralPath (Join-Path $PSScriptRoot '.preview.pid') -Value $previewProcess.Id
for ($attempt = 0; $attempt -lt 40; $attempt++) {
    try {
        $response = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/' -UseBasicParsing -TimeoutSec 1
        if ($response.StatusCode -eq 200) { Start-Process 'http://127.0.0.1:8000/'; Write-Host 'HALO preview: http://127.0.0.1:8000/'; exit 0 }
    } catch { Start-Sleep -Milliseconds 250 }
}
throw 'The preview server did not start. Run python preview_server.py to inspect the error.'
