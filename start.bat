@echo off
rem Startet start.py mit dem portablen Python dieses Projekts (runtime\python bzw. ..\..\runtime\python).
rem Gibt es ihn noch nicht, startet ein vorhandenes Python einmalig start.py, das den portablen Python dann herunterlaedt.
cd /d "%~dp0"
set "PP=runtime\python\python.exe"
if not exist "%PP%" set "PP=..\..\runtime\python\python.exe"
if exist "%PP%" goto run
where py >nul 2>nul && (py -3 start.py %* & goto end)
where python >nul 2>nul && (python start.py %* & goto end)
echo Kein Python gefunden. Bitte einmalig Python 3.10-3.12 installieren; start.py laedt danach den portablen Python.
goto end
:run
"%PP%" start.py %*
:end
echo.
pause
