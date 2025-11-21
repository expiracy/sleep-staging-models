from enum import Enum


class ModelType(Enum):
  PPG_UNFILTERED = 'ppg_unfiltered'
  PPG_UNFILTERED_WINDOWED = 'ppg_unfiltered_windowed'
  MULTIMODAL_CROSSATTN = 'multimodal_crossattn'
  MULTIMODAL_CROSSATTN_WINDOWED = 'multimodal_crossattn_windowed'
  PPG_ONLY = 'ppg_only'
  PPG_WITH_NOISE = 'ppg_with_noise'