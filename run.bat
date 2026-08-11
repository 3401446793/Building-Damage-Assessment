@echo off
chcp 65001 >nul
cd /d C:\Users\34014\Desktop\DIA

echo ========================================
echo 用嵌入式 Python 启动程序
echo ========================================

set PYTHONPATH=%~dp0python-3.10.10-embed-amd64\Lib\site-packages
set PATH=%~dp0python-3.10.10-embed-amd64;%~dp0python-3.10.10-embed-amd64\Scripts;%~dp0python-3.10.10-embed-amd64\DLLs;%~dp0python-3.10.10-embed-amd64\Lib;%PATH%

set GDAL_DATA=%~dp0python-3.10.10-embed-amd64\Library\share\gdal
set PROJ_LIB=%~dp0python-3.10.10-embed-amd64\Library\share\proj

.\python-3.10.10-embed-amd64\python.exe main.py

pause