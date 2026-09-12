@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title FreeToken 本地模型

set "HERE=%~dp0"
set "PY="
set "ARGS="

rem ---- 找 Python ----
where py >nul 2>&1 && (py -3 -c "import sys" >nul 2>&1 && set "PY=py -3")
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY if exist "C:\Program Files\Python310\python.exe" set "PY=C:\Program Files\Python310\python.exe"
if not defined PY where python >nul 2>&1 && set "PY=python"
if not defined PY (
  echo.
  echo [错误] 没有找到 Python。请安装 Python 3.10 或更高版本。
  echo.
  pause
  exit /b 1
)

rem ---- 参数：可以传模式名，也可以直接传模型目录 ----
if not "%~1"=="" (
  if /i "%~1"=="default"   set "ARGS=--mode default"
  if /i "%~1"=="low-mem"   set "ARGS=--mode low-mem"
  if /i "%~1"=="long-ctx"  set "ARGS=--mode long-ctx"
  if /i "%~1"=="high-conc" set "ARGS=--mode high-conc"
  if not defined ARGS      set "ARGS=--model "%~1""
)

echo.
echo ============================================================
echo    FreeToken 本地模型一键启动
echo ------------------------------------------------------------
echo    直接回车 = 自动找模型 + 自动算显存，默认端口 8421
echo    用法示例：
echo      %~nx0                      默认启动
echo      %~nx0 low-mem              省显存模式
echo      %~nx0 long-ctx             长上下文模式
echo      "%~nx0" "G:\AI\xxx-FTW"      指定模型目录
echo      %~nx0 --port 8088          换端口（后面可跟任意参数）
echo ============================================================
echo.

"%PY%" "%HERE%launch_model.py" %ARGS% %2 %3 %4 %5 %6 %7 %8 %9
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" (
  echo [退出码 %RC%] 启动失败，请看上方提示。
)
pause
