@echo off
setlocal enabledelayedexpansion
title FreeToken 转换工具

set "HERE=%~dp0"
set "PY="
set "PY32="

rem ===============================================================
rem  选 Python 解释器：优先 64 位。
rem
rem  坑：32 位 Python 跑在 64 位 Windows 上时，Windows 会把
rem  C:/Windows/System32 静默重定向到 SysWOW64，而那里既没有
rem  nvidia-smi.exe 也没有 nvml.dll。结果就是显卡明明在、命令行里
rem  nvidia-smi 也能跑，工具却报「未检测到 nvidia-smi」、显存显示 0，
rem  连 FreeToken 的显存预算都会因为读到 0 而判定装不下。
rem
rem  所以下面每一步都做位数校验，只认 64 位解释器。
rem  PY64CHK 故意用 assert 写法（不含括号），避免块内括号解析问题。
rem ===============================================================
set "PY64CHK=import sys;assert sys.maxsize//4294967296"

rem 1) py 启动器的 64 位入口（-64 后缀强制 64 位）
for %%V in (3.13 3.12 3.11 3.10) do (
  if not defined PY (
    py -%%V-64 -c "%PY64CHK%" >nul 2>&1
    if !errorlevel! equ 0 set "PY=py -%%V-64"
  )
)
if not defined PY (
  py -3-64 -c "%PY64CHK%" >nul 2>&1
  if !errorlevel! equ 0 set "PY=py -3-64"
)

rem 2) 常见 64 位安装目录
for %%P in (
  "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
  "C:/Program Files/Python313/python.exe"
  "C:/Program Files/Python312/python.exe"
  "C:/Program Files/Python311/python.exe"
  "C:/Program Files/Python310/python.exe"
) do (
  if not defined PY (
    if exist %%P (
      %%P -c "%PY64CHK%" >nul 2>&1
      if !errorlevel! equ 0 set PY=%%P
    )
  )
)

rem 3) PATH 里的 python（同样只认 64 位）
if not defined PY (
  python -c "%PY64CHK%" >nul 2>&1
  if !errorlevel! equ 0 set "PY=python"
)

rem 4) 实在只有 32 位：仍能跑（后端会自动改走 Sysnative 通道），但要明确提醒
if not defined PY (
  py -3 -c "import sys" >nul 2>&1
  if !errorlevel! equ 0 (
    set "PY=py -3"
    set "PY32=1"
  )
)

if not defined PY (
  echo.
  echo [错误] 没有找到 Python。请安装 64 位 Python 3.10 或更高版本后重试。
  echo.
  pause
  exit /b 1
)

if defined PY32 (
  echo.
  echo [警告] 只找到 32 位 Python，显卡识别可能不准。
  echo        建议安装 64 位 Python 后重新运行本脚本；
  echo        后端已做兼容，但 64 位才是正确做法。
  echo.
)

echo ============================================
echo   FreeToken 工具箱  (转换 + 运行 + 日志)
echo ============================================
echo   1. 浏览器会自动打开 http://127.0.0.1:8420
echo   2. 「格式转换」页: safetensors / GGUF 转成 FTW
echo   3. 「控制台」页: 直接启动大模型，实时看日志
echo.
echo   注意: 模型由本窗口托管，关闭窗口 = 停止模型
echo         并释放显存，不会留下占着显卡的孤儿进程。
echo ============================================
echo.

%PY% "%HERE%server.py" --port 8420

if errorlevel 1 (
  echo.
  echo [服务已退出] 如果端口被占用，可改用其它端口，例如：
  echo   %PY% "%HERE%server.py" --port 8421
  echo.
  pause
)
