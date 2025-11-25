# PowerShell training queue
# Trains both compressed and full windowed models

# Activate virtual environment
& C:\Users\james\Documents\repos\ppg-sleep-stage-classifier\src\.venv\Scripts\Activate.ps1

Set-Location C:\Users\james\Documents\repos\ppg-sleep-stage-classifier\src\sleep_staging_models

# python train_ppg_only.py --config configs/config_cloud.yaml

# TODO do this script with bigger window
# python train_ppg_unfiltered_windowed.py --config configs/config_windowed_sparse_window.yaml

python train_ppg_unfiltered_windowed.py --config configs/config_windowed_sdpaa.yaml 

# python train_ppg_unfiltered_windowed.py --config configs/config_windowed_sparse_window.yaml 

# python train_ppg_unfiltered_windowed.py --config configs/config_windowed_linear.yaml 

