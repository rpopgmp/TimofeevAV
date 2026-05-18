import numpy as np
import pandas as pd
import neurokit2 as nk
from tqdm.auto import tqdm
import warnings

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def basic_signal_features(signal):
    feats = {}
    for i, name in enumerate(LEAD_NAMES):
        x = signal[:, i]
        feats[f"{name}_mean"] = float(np.nanmean(x))
        feats[f"{name}_std"]  = float(np.nanstd(x))
        feats[f"{name}_rms"]  = float(np.sqrt(np.nanmean(x ** 2)))
        feats[f"{name}_ptp"]  = float(np.nanpercentile(x, 99) - np.nanpercentile(x, 1))
    return feats

def delineate(signal, fs=400.0, ref_lead=1):
    lead = signal[:, ref_lead].astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, info  = nk.ecg_peaks(lead, sampling_rate=fs,
                                 method="neurokit", correct_artifacts=True)
        rpeaks   = info["ECG_R_Peaks"]
        if len(rpeaks) < 3:
            raise RuntimeError(f"too few R-peaks: {len(rpeaks)}")
        _, waves = nk.ecg_delineate(lead, rpeaks, sampling_rate=fs, method="dwt")
    out = {"ECG_R_Peaks": list(rpeaks)}
    out.update({k: list(v) for k, v in waves.items()})
    return out

def _aggregate(values):
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=np.float64)
    if len(arr) == 0:
        return {"min": np.nan, "max": np.nan, "avg": np.nan}
    return {"min": float(arr.min()), "max": float(arr.max()), "avg": float(arr.mean())}


def _safe_get(peaks, key):
    return [int(i) for i in peaks.get(key, [])
            if i is not None and not (isinstance(i, float) and np.isnan(i)) and i >= 0]


def _per_beat_amplitude(signal, indices, lead, window_ms=20, fs=400.0):
    half, n, out = int(round(window_ms / 1000.0 * fs)), signal.shape[0], []
    for i in indices:
        if i >= n:
            continue
        a, b = max(0, i - half), min(n, i + half + 1)
        seg = signal[a:b, lead]
        if len(seg):
            out.append(signal[a + int(np.argmax(np.abs(seg))), lead])
    return out


def _per_beat_st_level(signal, r_offsets, lead, fs, offset_sec=0.08):
    off = int(round(offset_sec * fs))
    return [signal[ro + off, lead] for ro in r_offsets if 0 <= ro + off < signal.shape[0]]


def _interval_durations(starts, ends, fs):
    return [(b - a) / fs for a, b in zip(starts, ends)
            if a is not None and b is not None and a >= 0 and b >= 0 and b > a]


def morphology_features(signal, peaks, fs=400.0):
    n_samples, n_leads = signal.shape
    r_idx  = _safe_get(peaks, "ECG_R_Peaks")
    q_idx  = _safe_get(peaks, "ECG_Q_Peaks")
    s_idx  = _safe_get(peaks, "ECG_S_Peaks")
    p_idx  = _safe_get(peaks, "ECG_P_Peaks")
    t_idx  = _safe_get(peaks, "ECG_T_Peaks")
    r_on   = _safe_get(peaks, "ECG_R_Onsets")
    r_off  = _safe_get(peaks, "ECG_R_Offsets")
    p_on   = _safe_get(peaks, "ECG_P_Onsets")
    p_off  = _safe_get(peaks, "ECG_P_Offsets")
    t_on   = _safe_get(peaks, "ECG_T_Onsets")
    t_off  = _safe_get(peaks, "ECG_T_Offsets")

    feats = {}
    for name, vals in [
        ("QRS_duration", _interval_durations(r_on, r_off, fs)),
        ("PR_interval",  _interval_durations(p_on, r_on,  fs)),
        ("QT_interval",  _interval_durations(r_on, t_off, fs)),
        ("P_duration",   _interval_durations(p_on, p_off, fs)),
        ("T_duration",   _interval_durations(t_on, t_off, fs)),
    ]:
        for stat, v in _aggregate(vals).items():
            feats[f"{name}_{stat}"] = v

    for lead in range(n_leads):
        ln = LEAD_NAMES[lead]
        r_amp = _per_beat_amplitude(signal, r_idx, lead, fs=fs)
        q_amp = _per_beat_amplitude(signal, q_idx, lead, fs=fs)
        s_amp = _per_beat_amplitude(signal, s_idx, lead, fs=fs)
        p_amp = _per_beat_amplitude(signal, p_idx, lead, fs=fs)
        t_amp = _per_beat_amplitude(signal, t_idx, lead, fs=fs)
        qrs_amp = [signal[on:off, lead].max() - signal[on:off, lead].min()
                   for on, off in zip(r_on, r_off) if 0 <= on < off <= n_samples and off - on > 0]
        rs_ratio = [r_amp[k] / abs(s_amp[k]) for k in range(min(len(r_amp), len(s_amp))) if s_amp[k] != 0]
        st_level = _per_beat_st_level(signal, r_off, lead, fs)
        for name, vals in [("R_amp", r_amp), ("Q_amp", q_amp), ("S_amp", s_amp),
                            ("P_amp", p_amp), ("T_amp", t_amp), ("QRS_amplitude", qrs_amp),
                            ("R_S_ratio", rs_ratio), ("ST_level", st_level)]:
            for stat, v in _aggregate(vals).items():
                feats[f"{name}_{ln}_{stat}"] = v
    return feats


def extract_features(signal, fs=400.0, ref_lead=1):
    feats = basic_signal_features(signal)
    waves = delineate(signal, fs=fs, ref_lead=ref_lead)
    feats.update(morphology_features(signal, waves, fs=fs))
    return feats


def extract_features_dataset(df, signals, fs=400.0, ref_lead=1, idx_col="signal_idx"):
    basic_rows, full_rows = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="features"):
        signal = np.array(signals[int(row[idx_col])]).astype(np.float32)
        basic = basic_signal_features(signal)
        basic["signal_idx"] = int(row[idx_col])
        basic_rows.append(basic)
        try:
            full = extract_features(signal, fs=fs, ref_lead=ref_lead)
            full["delineation_ok"] = True
        except Exception:
            full = dict(basic)
            full["delineation_ok"] = False
        full["signal_idx"] = int(row[idx_col])
        full_rows.append(full)

    n_failed = sum(1 for r in full_rows if not r["delineation_ok"])
    if n_failed:
        print(f"delineation failed: {n_failed}/{len(df)} ({100*n_failed/len(df):.1f}%)")
    return pd.DataFrame(basic_rows), pd.DataFrame(full_rows)
