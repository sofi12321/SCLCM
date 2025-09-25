import numpy as np
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb
import matplotlib.colors as mcolors


# Labels / mappings
freq_labels = ["Delta (1-4Hz)",
               "Theta (4-8Hz)",
               "Alpha (8-13Hz)",
               "Beta (13-31Hz)",
               "Gamma (31-50Hz)"]


# Edit if your downstream task uses different class ids
class_label_mapping = {
    0: "Negative Emotion",
    1: "Neutral Emotion",
    2: "Positive Emotion"
}


# Color scale
vmin_fixed = -1e-5
vmax_fixed =  1e-5
common_norm = mcolors.TwoSlopeNorm(vmin=vmin_fixed, vcenter=0.0, vmax=vmax_fixed)


# Custom diverging colormap based on hue (same as your notebook)
hue_neg, hue_zero, hue_pos = 0.18, 0.29, 0.05
color_neg  = hsv_to_rgb((hue_neg,  1, 1))
color_zero = hsv_to_rgb((hue_zero, 1, 1))
color_pos  = hsv_to_rgb((hue_pos,  1, 1))
common_cmap = LinearSegmentedColormap.from_list(
    "custom_hue",
    [(0.0, color_zero),
     (0.5, color_neg),
     (1.0, color_pos)]
)


# Canonical channel name lists (SEED/DEAP orderings you use)
SEED_62 = np.array([
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ',
    'C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
    'P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4',
    'PO6','PO8','CB1','O1','OZ','O2','CB2'
])

DEAP_32 = np.array([
    'FP1', 'FP2', 'AF4', 'AF3', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC6',
    'FC2', 'FC1', 'FC5', 'T7', 'C3', 'CZ', 'C4', 'T8', 'CP6', 'CP2',
    'CP1', 'CP5', 'P7', 'P3', 'PZ', 'P4', 'P8', 'PO4', 'PO3', 'O1',
    'OZ', 'O2'
])



# 2D custom montage positions (x, y, z=0) 
CUSTOM_POS_2D = {
    "FP1": (-0.025,  0.10), "FPZ": ( 0.00,  0.11), "FP2": ( 0.025, 0.10),
    "AF3": (-0.055,  0.085), "AF4": ( 0.055, 0.085),
    "F7":  (-0.08,   0.07),  "F5":  (-0.06,  0.065), "F3": (-0.04, 0.062),
    "F1":  (-0.02,   0.061), "FZ":  ( 0.00, 0.060), "F2": ( 0.02, 0.061),
    "F4":  ( 0.04,   0.062), "F6":  ( 0.06, 0.065), "F8": ( 0.08, 0.070),
    "FT7": (-0.10,   0.04),  "FC5": (-0.075, 0.035), "FC3": (-0.05, 0.032),
    "FC1": (-0.025,  0.03),  "FCZ": ( 0.000, 0.028), "FC2": ( 0.025, 0.03),
    "FC4": ( 0.05,   0.032), "FC6": ( 0.075, 0.035), "FT8": ( 0.10, 0.04),
    "T7":  (-0.11,   0.00),  "C5":  (-0.083, 0.00),  "C3": (-0.055, 0.00),
    "C1":  (-0.027,  0.00),  "CZ":  ( 0.000, 0.00),  "C2": ( 0.027, 0.00),
    "C4":  ( 0.055,  0.00),  "C6":  ( 0.083, 0.00),  "T8": ( 0.11,  0.00),
    "TP7": (-0.10,  -0.04),  "CP5": (-0.075,-0.035), "CP3": (-0.05,-0.032),
    "CP1": (-0.025, -0.03),  "CPZ": ( 0.000,-0.028), "CP2": ( 0.025,-0.03),
    "CP4": ( 0.05,  -0.032), "CP6": ( 0.075,-0.035), "TP8": ( 0.10,-0.04),
    "P7":  (-0.08,  -0.07),  "P5":  (-0.06, -0.065), "P3": (-0.04,-0.062),
    "P1":  (-0.02,  -0.061), "PZ":  ( 0.00, -0.060), "P2": ( 0.02,-0.061),
    "P4":  ( 0.04,  -0.062), "P6":  ( 0.06, -0.065), "P8": ( 0.08,-0.070),
    "PO7": (-0.062, -0.089), "PO5": (-0.041,-0.087), "PO3": (-0.02,-0.085),
    "POZ": ( 0.000, -0.083), "PO4": ( 0.02, -0.085), "PO6": ( 0.041,-0.087),
    "PO8": ( 0.062, -0.089), "CB1": (-0.040,-0.103), "O1": (-0.02,-0.108),
    "OZ":  ( 0.000, -0.110), "O2":  ( 0.02, -0.108), "CB2": ( 0.04,-0.103),
}


def pick_channel_names(all_channel_names, ch_indices):
    """Helper to pick the exact channel names that were fed into the model."""
    return [all_channel_names[i] for i in ch_indices]
