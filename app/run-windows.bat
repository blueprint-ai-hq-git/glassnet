@echo off
REM GlassNet Runner on Windows: a local window, no server, no internet.
setlocal
set HERE=%~dp0
set PAGE=file:///%HERE:\=/%index.html
set CHROME=
for %%P in ("%ProgramFiles%\Google\Chrome\Application\chrome.exe" "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" "%LocalAppData%\Google\Chrome\Application\chrome.exe" "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe" "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe") do (
  if exist %%P set CHROME=%%~P
)
if "%CHROME%"=="" (
  echo No Chrome or Edge found. Opening with the default browser instead.
  start "" "%HERE%index.html"
) else (
  start "" "%CHROME%" --app="%PAGE%" --window-size=1180,880 --user-data-dir="%LocalAppData%\GlassNetRunner"
)
