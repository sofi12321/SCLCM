from typing import Dict, List, Sequence, Mapping
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mne


from utils.xai_config import (
    freq_labels, class_label_mapping,
    vmin_fixed, vmax_fixed, common_cmap, common_norm,
)



def _avg_heatmap(data_list: List[np.ndarray]) -> np.ndarray:
    """
    data_list: list of arrays with shape (F, C, T)
    returns (F, C) after avg over samples and time.
    """
    arr = np.stack(data_list, axis=0)        # (N, F, C, T)
    return arr.mean(axis=-1).mean(axis=0)    # (F, C)


def _build_info(channel_names: Sequence[str], ch_pos_map: Mapping[str, Sequence[float]]):
    """
    channel_names: exactly what went into the model, in that order.
    ch_pos_map: dict{name: (x,y) or (x,y,z)}
    Returns an MNE Info with a custom montage.
    """
    # Allow (x,y) or (x,y,z). If (x,y), set z=0
    ch_pos = {}
    for ch in channel_names:
        xy = ch_pos_map[ch]
        if len(xy) == 2:
            ch_pos[ch] = (xy[0], xy[1], 0.0)
        else:
            ch_pos[ch] = tuple(xy)

    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    info = mne.create_info(ch_names=list(channel_names), sfreq=250.0,
                           ch_types=['eeg'] * len(channel_names))
    info.set_montage(montage, match_case=False, on_missing='ignore')
    return info


def _plot_topomap(ch_values: np.ndarray, info, title: str):
    """
    ch_values: (C,) values aligned with info.ch_names order.
    """
    pos = np.array([info['chs'][i]['loc'][:3] for i in range(len(info['ch_names']))])[:, :2]
    valid = ~np.isnan(pos).any(axis=1)

    from mne.viz import plot_topomap
    fig, ax = plt.subplots()
    im, _ = plot_topomap(
        ch_values[valid], pos[valid],
        axes=ax, show=False, cmap=common_cmap,
        vlim=(vmin_fixed, vmax_fixed), sensors=True,
        contours=3, extrapolate="head", sphere=(0,0,0,0.125),
        names=np.array(info.ch_names)[valid]
    )
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    plt.show()


def summarize_attributions(
    attributions_by_class: Dict[int, List[np.ndarray]],
    channel_names: Sequence[str],
    ch_pos_map: Mapping[str, Sequence[float]],
    use_frequency: bool = True,
):
    """
    Makes class-wise heatmaps of (freq x channels) and per-frequency topomaps.

    Parameters
    ----------
    attributions_by_class : dict[int, list[np.ndarray]]
        Values are arrays of shape (F, C, T).
    channel_names : list[str]
        Names in the exact order that went into the model.
    ch_pos_map : dict[str, (x, y) or (x, y, z)]
        Channel-name -> 2D/3D position. (z will be set to 0 if absent.)
    use_frequency : bool
        If False, we still treat F=1 for heatmap; no per-band loop is shown.
    """
    info = _build_info(channel_names, ch_pos_map)

    for cls, samples in attributions_by_class.items():
        label = class_label_mapping.get(cls, str(cls))

        # Heatmap (F x C)
        avg_fc = _avg_heatmap(samples)      # (F, C)
        plt.figure(figsize=(10, 3 + 0.2*len(channel_names)))
        plt.imshow(avg_fc, aspect="auto", cmap=common_cmap, norm=common_norm)
        plt.colorbar()
        plt.title(f"Average Relevance Heatmap — {label}")
        plt.xlabel("Channels")
        plt.ylabel("Frequency Band" if use_frequency else "Band")
        plt.xticks(ticks=np.arange(len(channel_names)), labels=channel_names, rotation=90)
        if use_frequency and avg_fc.shape[0] == len(freq_labels):
            plt.yticks(ticks=np.arange(len(freq_labels)), labels=freq_labels)
        plt.tight_layout()
        plt.show()

        # Bar per-channel (aggregated over freq/time)
        ch_vals = avg_fc.mean(axis=0)       # (C,)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(np.arange(len(ch_vals)), ch_vals, color=common_cmap(common_norm(ch_vals)))
        ax.set_title(f"Avg. Channel Relevance — {label}")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Avg. Relevance")
        ax.set_xticks(np.arange(len(channel_names)))
        ax.set_xticklabels(channel_names, rotation=90)
        sm = plt.cm.ScalarMappable(norm=common_norm, cmap=common_cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax)
        fig.tight_layout()
        plt.show()

        # Per-frequency topomaps (if F>1)
        if use_frequency and avg_fc.shape[0] > 1:
            for f_idx in range(avg_fc.shape[0]):
                band_name = freq_labels[f_idx] if f_idx < len(freq_labels) else f"Band {f_idx}"
                _plot_topomap(avg_fc[f_idx, :], info, title=f"{label} — {band_name}")

        # Aggregated topomap (all frequencies)
        _plot_topomap(avg_fc.mean(axis=0), info, title=f"{label} — Aggregated")
