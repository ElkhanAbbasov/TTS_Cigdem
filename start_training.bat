@echo off
echo ========================================
echo Starting Cigdem TTS Training
echo ========================================
echo.
echo Configuration:
echo - Checkpoint: Fresh start from pretrained
echo - Batch size: 2 (will fallback to 1 if OOM)
echo - Epochs: 100
echo - Save frequency: Every 2 epochs
echo.
echo Press Ctrl+C to stop training
echo ========================================
echo.

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.\venv\Scripts\python.exe train_finetune.py --config_path Configs/config_ft.yml

if errorlevel 1 (
    echo.
    echo ========================================
    echo Training failed! Check errors above.
    echo ========================================
    pause
) else (
    echo.
    echo ========================================
    echo Training completed successfully!
    echo ========================================
    pause
)
