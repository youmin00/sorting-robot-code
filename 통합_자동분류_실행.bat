@echo off
chcp 65001 > nul
setlocal

set "ROOT=%~dp0"
set "PY=%ROOT%.venv312\Scripts\python.exe"
set "SCRIPT=%ROOT%통합_자동분류_실행.py"

if not exist "%PY%" (
    echo [ERROR] Not found: "%PY%"
    echo Please create/install .venv312 first.
    pause
    exit /b 1
)

if not exist "%SCRIPT%" (
    echo [ERROR] Not found: "%SCRIPT%"
    pause
    exit /b 1
)

"%PY%" "%SCRIPT%" %*
set "EC=%ERRORLEVEL%"

if not "%EC%"=="0" (
    echo.
    echo [ERROR] Camera script exited with code %EC%.
    pause
)

exit /b %EC%
