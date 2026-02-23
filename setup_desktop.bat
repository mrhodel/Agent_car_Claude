@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  Desktop training setup for Yahboom Raspbot V2 Agent
REM  Run once from the project root:  setup_desktop.bat
REM
REM  Requirements: Python 3.10-3.12, NVIDIA GPU with CUDA 12.1+
REM  Check your CUDA version: nvidia-smi
REM ─────────────────────────────────────────────────────────────────────────

echo === Creating virtual environment ===
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo === Installing PyTorch with CUDA 12.1 ===
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo.
echo === Installing other dependencies ===
pip install -r requirements_desktop.txt

echo.
echo === Verifying install ===
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo === Setup complete ===
echo.
echo To train run:
echo   venv\Scripts\activate.bat
echo   python main.py --mode train --episodes 1500
echo.
echo To transfer checkpoint to Pi:
echo   scp models\checkpoints\policy_ep1500.pt pi@PI_IP_ADDRESS:~/robot_car_claude/Agent_car_Claude/models/checkpoints/
pause
