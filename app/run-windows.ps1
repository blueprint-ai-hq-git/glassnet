# GlassNet Runner on Windows: opens the app from this folder. No server, no internet, no install.
Start-Process (Join-Path $PSScriptRoot "index.html")
