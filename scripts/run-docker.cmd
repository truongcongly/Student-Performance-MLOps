@echo off
setlocal
cd /d "%~dp0.."

docker info >nul 2>&1
if errorlevel 1 (
    echo Docker daemon is not available. Start Docker Desktop and try again.
    exit /b 1
)

docker build -t student-performance-api:local .
if errorlevel 1 exit /b 1

docker run --rm --detach --name student-performance-api --publish 127.0.0.1:8001:8000 student-performance-api:local
if errorlevel 1 exit /b 1

echo API: http://127.0.0.1:8001/docs
