@echo off
:: 本文件以 GBK(936) 编码保存：中文 Windows 的 cmd 默认代码页就是 936，
:: 若改用 UTF-8 文件 + chcp 65001，cmd 逐行读取时会在切换解码器的位置
:: 把多字节汉字切断，导致整行被当成命令报错（实测过）。
chcp 936 >nul
setlocal EnableDelayedExpansion

:: --- 自动提权（diskpart 需要管理员权限） ---
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo 需要管理员权限，正在提权...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

echo ============================================
echo   WSL2 / Docker 数据盘压缩工具
echo ============================================
echo.

:: --- 自动发现所有 vhdx ---
:: 三个常见位置：Docker 数据盘、docker-desktop 发行版、商店版发行版（Debian 等）
:: -Depth 4 足以覆盖 Packages\<发行版>\LocalState\ext4.vhdx 这种层级
echo 正在查找数据盘...
set "IDX=0"
for /f "usebackq tokens=1,2 delims=|" %%A in (`powershell -NoProfile -Command "$r=@('%LOCALAPPDATA%\Docker','%LOCALAPPDATA%\wsl','%LOCALAPPDATA%\Packages'); foreach($d in $r){ if(Test-Path $d){ Get-ChildItem -LiteralPath $d -Recurse -Depth 4 -Filter *.vhdx -ErrorAction SilentlyContinue | ForEach-Object { '{0}|{1}' -f $_.FullName, [math]::Round($_.Length/1GB,2) } } }"`) do (
    set /a IDX+=1
    set "VHDX_!IDX!=%%A"
    set "BEFORE_!IDX!=%%B"
)
set "TOTAL=!IDX!"

if !TOTAL!==0 (
    echo.
    echo [错误] 未找到任何 vhdx 数据盘。
    echo        如果 Docker Desktop 装在非默认位置，请手动修改本脚本中的查找路径。
    echo.
    pause
    exit /b 1
)

echo.
echo 找到 !TOTAL! 个数据盘：
echo.
for /l %%I in (1,1,!TOTAL!) do (
    echo    [%%I] !BEFORE_%%I! GB
    echo        !VHDX_%%I!
)
echo.
echo 即将执行：
echo    1. 关闭 Docker Desktop 与 WSL
echo    2. 逐个压缩上述数据盘（16GB 约需 1-3 分钟）
echo.
echo 警告：WSL 会被强制关闭，其中未保存的工作会丢失。
echo       请先关掉 WSL 里的终端、编辑器等程序。
echo.
set /p "CONFIRM=继续？输入 Y 回车，其他键取消："
if /i not "!CONFIRM!"=="Y" (
    echo.
    echo 已取消，未做任何改动。
    pause
    exit /b 0
)

:: --- 关闭占用 ---
echo.
echo [1/3] 关闭 Docker Desktop...
taskkill /IM "Docker Desktop.exe" /F >nul 2>&1
taskkill /IM "com.docker.backend.exe" /F >nul 2>&1
:: 用 ping 而不是 timeout 作延时：timeout 在 stdin 被重定向时会直接报错退出
ping -n 3 127.0.0.1 >nul

echo [2/3] 关闭 WSL...
wsl --shutdown
ping -n 6 127.0.0.1 >nul

:: --- 逐个压缩 ---
:: diskpart 不支持命令行传参，只能读脚本文件（不要用管道喂它：
:: 管道左侧的括号块里延迟扩展会失效，变量不会被展开）
echo [3/3] 压缩数据盘...
set "DP_SCRIPT=%TEMP%\compact_vhdx.txt"
for /l %%I in (1,1,!TOTAL!) do (
    echo.
    echo   --- [%%I/!TOTAL!] !VHDX_%%I! ---
    >  "!DP_SCRIPT!" echo select vdisk file="!VHDX_%%I!"
    >> "!DP_SCRIPT!" echo attach vdisk readonly
    >> "!DP_SCRIPT!" echo compact vdisk
    >> "!DP_SCRIPT!" echo detach vdisk
    diskpart /s "!DP_SCRIPT!"
)
del "!DP_SCRIPT!" >nul 2>&1

:: --- 压缩后大小 ---
echo.
echo ============================================
echo   压缩结果
echo ============================================
echo.
for /l %%I in (1,1,!TOTAL!) do (
    for /f "usebackq" %%A in (`powershell -NoProfile -Command "[math]::Round((Get-Item -LiteralPath '!VHDX_%%I!').Length/1GB,2)"`) do set "AFTER_%%I=%%A"
    echo    !BEFORE_%%I! GB  -^>  !AFTER_%%I! GB
    echo        !VHDX_%%I!
)
echo.
echo 完成。请重新打开 Docker Desktop 继续使用。
echo.
echo 提示：Debian 发行版若想以后自动回收空间（不必每次手动压），
echo       可在 WSL 关闭时执行一次：
echo         wsl --manage Debian --set-sparse true --allow-unsafe
echo.
pause
