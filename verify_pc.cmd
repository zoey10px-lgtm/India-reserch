@echo off
setlocal
set "VERIFY_SCRIPT=%~dp0tools\verify_pc.py"
set "VERIFY_EXIT=0"
where py.exe >nul 2>&1
if errorlevel 1 goto use_python
py -3.12 -c "import sys; sys.exit(0)" >nul 2>&1
if errorlevel 1 goto use_python
py -3.12 "%VERIFY_SCRIPT%" %*
set "VERIFY_EXIT=%ERRORLEVEL%"
goto done

:use_python
where python.exe >nul 2>&1
if errorlevel 1 goto missing_python
python "%VERIFY_SCRIPT%" %*
set "VERIFY_EXIT=%ERRORLEVEL%"
goto done

:missing_python
echo Install 64-bit Python 3.12, then run verify_pc.cmd again.
set "VERIFY_EXIT=1"

:done
if "%~1"=="" pause
exit /b %VERIFY_EXIT%
