@echo off
setlocal

set "RELEASE_MPC_ROOT=%~dp0.."
for %%I in ("%RELEASE_MPC_ROOT%") do set "RELEASE_MPC_ROOT=%%~fI"
set "MPC_TEMP=%RELEASE_MPC_ROOT%\temp"
if not exist "%MPC_TEMP%" mkdir "%MPC_TEMP%"
set "TEMP=%MPC_TEMP%"
set "TMP=%MPC_TEMP%"

call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%

rem Collapse duplicate PATH/Path entries exposed by the Codex launcher.
set "MPC_CLEAN_PATH=%PATH%"
set "Path="
set "PATH=%MPC_CLEAN_PATH%"

set "CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

"%CMAKE_EXE%" -S "%RELEASE_MPC_ROOT%\native\reduced_predictor" -B "%RELEASE_MPC_ROOT%\build\reduced_native" -G "Visual Studio 17 2022" -A x64
if errorlevel 1 exit /b %errorlevel%

"%CMAKE_EXE%" --build "%RELEASE_MPC_ROOT%\build\reduced_native" --config Release --target MPCJSBSim -- /m
if errorlevel 1 exit /b %errorlevel%

echo Built: %RELEASE_MPC_ROOT%\runtime\predictor\Release\MPCJSBSim.dll
exit /b 0
