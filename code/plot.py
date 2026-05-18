import numpy as np
import matplotlib.pyplot as plt


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

PEAK_STYLES = {
    "ECG_P_Peaks":   {"color": "#1f77b4", "marker": "o", "label": "P"},
    "ECG_Q_Peaks":   {"color": "#2ca02c", "marker": "v", "label": "Q"},
    "ECG_R_Peaks":   {"color": "#d62728", "marker": "^", "label": "R"},
    "ECG_S_Peaks":   {"color": "#9467bd", "marker": "v", "label": "S"},
    "ECG_T_Peaks":   {"color": "#ff7f0e", "marker": "o", "label": "T"},
    "ECG_R_Onsets":  {"color": "#7f7f7f", "marker": "|", "label": "QRS onset"},
    "ECG_R_Offsets": {"color": "#7f7f7f", "marker": "|", "label": "QRS offset"},
    "ECG_P_Onsets":  {"color": "#aec7e8", "marker": "|", "label": "P onset"},
    "ECG_P_Offsets": {"color": "#aec7e8", "marker": "|", "label": "P offset"},
    "ECG_T_Onsets":  {"color": "#ffbb78", "marker": "|", "label": "T onset"},
    "ECG_T_Offsets": {"color": "#ffbb78", "marker": "|", "label": "T offset"},
}


def plot_ecg(signal, peaks=None, fs=400.0, title=None):
    signal = np.asarray(signal)
    n_samples, n_leads = signal.shape
    t = np.arange(n_samples) / fs

    fig, axes = plt.subplots(n_leads, 1, sharex=True,
                             figsize=(14, max(3.0, 0.9 * n_leads)),
                             constrained_layout=True)
    if n_leads == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        ax.plot(t, signal[:, idx], lw=0.8, color="#1f77b4")
        ax.grid(True, alpha=0.25)
        ax.margins(y=0.18)
        ax.tick_params(axis="both", labelsize=8)
        ax.set_ylabel(LEAD_NAMES[idx] if idx < len(LEAD_NAMES) else f"L{idx}",
                      rotation=0, labelpad=20, fontsize=9)

        if peaks is not None:
            for key, style in PEAK_STYLES.items():
                idxs = peaks.get(key, [])
                idxs = [int(i) for i in idxs
                        if i is not None and not (isinstance(i, float) and np.isnan(i))
                        and 0 <= i < n_samples]
                if not idxs:
                    continue
                ax.scatter(np.asarray(idxs) / fs, signal[idxs, idx],
                           s=30, c=style["color"], marker=style["marker"],
                           label=style["label"] if idx == 0 else None, zorder=5)

    if peaks is not None:
        axes[0].legend(loc="upper right", ncol=6, fontsize=7, framealpha=0.9)
    axes[-1].set_xlabel("Time, sec")
    if title is not None:
        fig.suptitle(title, fontsize=14)
    plt.show()
    return fig, axes
