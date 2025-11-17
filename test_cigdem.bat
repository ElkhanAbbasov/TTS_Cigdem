@echo off
echo ========================================
echo Cigdem TTS - Quick Test
echo ========================================
echo.

REM Check if checkpoint directory exists
if not exist "Models\Cigdem_TTS" (
    echo ERROR: Models\Cigdem_TTS directory not found!
    echo.
    echo Please create it and copy your epoch files:
    echo   mkdir Models\Cigdem_TTS
    echo   copy your-downloads\epoch_*.pth Models\Cigdem_TTS\
    echo.
    pause
    exit /b
)

REM List available checkpoints
echo Available checkpoints:
dir /b Models\Cigdem_TTS\epoch_*.pth 2>nul
if errorlevel 1 (
    echo No checkpoints found!
    echo Please copy your epoch_*.pth files to Models\Cigdem_TTS\
    echo.
    pause
    exit /b
)

echo.
echo ========================================
echo Starting interactive testing...
echo ========================================
echo.

python inference_local.py

pause
