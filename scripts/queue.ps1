# PowerShell training queue
# Trains both compressed and full windowed models

# Activate virtual environment
& C:\Users\james\Documents\repos\ppg-sleep-stage-classifier\src\.venv\Scripts\Activate.ps1

Set-Location C:\Users\james\Documents\repos\ppg-sleep-stage-classifier\src\sleep_staging_models

python train_ppg_unfiltered.py --config configs/config_ppg_unfiltered.yaml --runs 5
