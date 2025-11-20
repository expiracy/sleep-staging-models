# PowerShell training queue
# Trains both compressed and full windowed models

# Activate virtual environment
& C:\Users\james\Documents\repos\ppg-sleep-stage-classifier\src\.venv\Scripts\Activate.ps1

Set-Location C:\Users\james\Documents\repos\ppg-sleep-stage-classifier\src\sleep_staging_models


python train_windowed_crossattn.py --config configs/config_windowed.yaml --runs 1

