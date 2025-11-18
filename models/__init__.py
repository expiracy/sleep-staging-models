"""
Model architectures for sleep staging
"""
from .multimodal_sleep_model import SleepPPGNet, MultiModalSleepNet
from .multimodal_model_crossattn import ImprovedMultiModalSleepNet
from .ppg_unfiltered_crossattn import PPGUnfilteredCrossAttention
from .ppg_with_noise_baseline import PPGWithNoiseBaseline

__all__ = [
    'SleepPPGNet',
    'MultiModalSleepNet', 
    'ImprovedMultiModalSleepNet',
    'PPGUnfilteredCrossAttention',
    'PPGWithNoiseBaseline',
]
