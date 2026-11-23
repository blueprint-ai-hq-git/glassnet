# Windows: serve the app and open it. Needs only Python 3.
cd $PSScriptRoot
$port = if ($env:PORT) { $env:PORT } else { 8781 }
Start-Process -NoNewWindow python -ArgumentList "-m","http.server",$port,"--bind","127.0.0.1"
Start-Sleep -Seconds 1
Start-Process "http://127.0.0.1:$port/index.html"
