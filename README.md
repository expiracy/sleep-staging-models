# On Improving PPG-Based Sleep Staging: A Pilot Study

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of "On Improving PPG-Based Sleep Staging: A Pilot Study" - exploring dual-stream cross-attention architectures for enhanced sleep stage classification using PPG signals.

## 📋 Abstract

This repository contains the implementation of our pilot study on improving PPG-based sleep staging through dual-stream cross-attention architectures. We demonstrate that substantial performance gains can be achieved by combining PPG with auxiliary modalities (Augmented PPG, Synthetic ECG, Real ECG) under a dual-stream cross-attention framework, achieving up to **83.3% accuracy** and **0.745 Cohen's kappa** on the MESA dataset.

## 🔑 Key Findings

- **PPG + Augmented PPG** achieves the best performance (κ=0.745, Acc=83.3%), improving accuracy by 5% over single-stream baseline
- Cross-attention mechanism effectively extracts complementary information from signal variations
- Augmented PPG strategy performs comparably to PPG + Real ECG, while being more practical (no additional sensors needed)
- Synthetic ECG shows promise but requires sleep-specific training for optimal performance

## 🏗️ Architecture Overview

### Single-Stream Model (Baseline)
- **SleepPPG-Net**: Processes 10-hour PPG recordings through residual convolutional blocks
- Achieves κ=0.675, Accuracy=78.3% on MESA test set

### Dual-Stream Models
1. **PPG + Augmented PPG**: Combines clean and noise-augmented PPG signals
2. **PPG + Synthetic ECG**: Uses RDDM-generated ECG as auxiliary modality
3. **PPG + Real ECG**: Upper bound performance using actual ECG recordings

<div align="center">
  <img src="docs/dual_stream_architecture.png" alt="Dual-Stream Architecture" width="600"/>
</div>

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/DavyWJW/sleep-staging-models.git
cd sleep-staging-models

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.9.0
- NumPy ≥ 1.19.0
- scikit-learn ≥ 0.24.0
- h5py ≥ 3.0.0
- See `requirements.txt` for complete list

### Data Preparation

1. Download the <a target="_blank" href="https://sleepdata.org/datasets/mesa/files/polysomnography" data-view-component="true" class="Link mr-2">polysomnography</a> data of MESA Sleep Study dataset.

2. Data Processing: Extract PPG and ECG data from MESA data.
```bash
python extract_mesa_data.py 
```

## 🏃‍♂️ Training

### Train Single-Stream Baseline (SleepPPG-Net)

```bash
python train_cloud.py --config configs/config_cloud.yaml --model ppg_only --runs 5
```

### Train Dual-Stream Models

```bash
# PPG + Augmented PPG (Best Performance)
python train_ppg_unfiltered.py --config configs/config_ppg_unfiltered.yaml --runs 5

# PPG + Synthetic ECG
python train_crossattn_gen.py --config configs/config_crossattn_generated.yaml --model_type generated_ecg --runs 5

# PPG + Real ECG
python train_crossattn.py --config configs/config_crossattn_v2.yaml --runs 5
```

### Multi-GPU Training (DDP)

```bash
python train_crossattn_gen.py --config configs/config_crossattn_generated.yaml --gpus 3 --runs 5
```

## 📝 Citation

If you find this work useful for your research, please kindly cite these by:

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{ppg_sleep_staging_2025,
  title={On Improving PPG-Based Sleep Staging: A Pilot Study},
  author={[Author Names]},
  booktitle={Conference Name},
  year={2025},
  pages={1--4},
  doi={10.1145/XXXXXXX.XXXXXXX}
}
```

## 🤝 Acknowledgments

- MESA Sleep Study dataset [14]
- SleepPPG-Net baseline architecture [9]
- RDDM for synthetic ECG generation [11]

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or collaborations, please:
- Open an issue on GitHub
- Email: [corresponding.author@email.com]

## 🔮 Future Work

- Investigate other auxiliary modalities (e.g., EEG)
- Evaluate on additional sleep staging datasets
- Develop sleep-specific synthetic ECG generation
- Explore lightweight architectures for edge deployment

---

<p align="center">
<i>Advancing accessible sleep monitoring through innovative computational approaches</i>
</p>
