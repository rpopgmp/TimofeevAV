import tarfile
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import wfdb
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, peak_prominences, resample_poly, wiener
from tqdm.auto import tqdm


def unify_ecg(signal, fs):
    signal, fs = np.asarray(signal), int(fs)
    if fs != 400:
        g = gcd(fs, 400)
        signal = resample_poly(signal, up=400 // g, down=fs // g, axis=0)
    return signal[:2800, :].astype(np.float32)




def filter_signal(signal):
    signal = np.asarray(signal, dtype=np.float32)
    n_samples, n_leads = signal.shape
    dx_all, mdx_all = np.zeros_like(signal), np.zeros_like(signal)
    for li in range(n_leads):
        ca1, cd1 = pywt.dwt(signal[:, li], "coif4")
        sigma = np.median(np.abs(cd1)) / 0.6745
        cd1 = np.where(np.abs(cd1) > sigma * np.sqrt(2 * np.log(len(signal[:, li]))), cd1, 0.0)
        ca1 = np.where(np.isfinite(w := wiener(ca1, mysize=15)), w, 0.0)
        dx = pywt.idwt(ca1, cd1, "coif4")[:n_samples]
        if len(dx) < n_samples:
            dx = np.pad(dx, (0, n_samples - len(dx)), mode="edge")
        dx_all[:, li] = dx.astype(np.float32)
        mdx_all[:, li] = median_filter(dx, size=8).astype(np.float32)

    ref = dx_all[:, 0]
    prom = 0.5 * np.std(ref)
    peaks = np.sort(np.concatenate([
        find_peaks(ref,  distance=80, prominence=prom)[0],
        find_peaks(-ref, distance=80, prominence=prom)[0],
    ]))
    restored = mdx_all.copy()
    for p in peaks:
        restored[max(0, p-8):min(n_samples, p+9), :] = dx_all[max(0, p-8):min(n_samples, p+9), :]
    return restored


def preprocess_record(record_path):
    signal, fields = wfdb.rdsamp(str(record_path))
    return filter_signal(unify_ecg(signal, fs=int(fields["fs"])))


def augment_ecg(ecg, rho=0.6, f_min=0.5, f_max=1.5, delta=20, distance=150, prom_quantile=0.7):
    ecg, T, C = ecg.copy(), *ecg.shape
    lead0 = ecg[:, 0]
    peaks, _ = find_peaks(lead0, distance=distance)
    if not len(peaks):
        return ecg
    prom = peak_prominences(lead0, peaks)[0]
    peaks = peaks[prom >= np.quantile(prom, prom_quantile)]
    for peak in peaks:
        if np.random.rand() > rho:
            continue
        f = np.random.uniform(f_min, f_max)
        for ch in range(C):
            sig, base = ecg[:, ch].copy(), np.mean(ecg[:, ch])
            lb, rb = max(0, peak-delta), min(T, peak+delta)
            local = sig[lb:rb] - base
            if not len(local):
                continue
            lp = lb + int(np.argmax(np.abs(local)))
            left = lp
            while left > max(0, lp-delta) and (sig[left]-base)*(sig[lp]-base) > 0:
                left -= 1
            right = lp
            while right < min(T-1, lp+delta) and (sig[right]-base)*(sig[lp]-base) > 0:
                right += 1
            seg = sig[left:right]
            if not len(seg):
                continue
            scaled = base + (seg - base) * f
            L, pp = len(seg), lp - left
            half = max(pp, L - 1 - pp, 1)
            x = np.linspace(-pp/half, (L-1-pp)/half, L)
            smooth = np.exp(-(x**2) / 0.18)
            smooth /= smooth.max()
            ecg[left:right, ch] = base + (scaled - base) * smooth
    return ecg


def filter_records(df, age_min=0, age_max=120):
    df = df.copy()
    ok = df["duration_sec"].ge(7) & df["age"].between(age_min, age_max) & df["age"].notna()
    print(f"filter_records: {ok.sum()}/{len(df)} kept")
    return df[ok].reset_index(drop=True)


def preprocess_dataset(df, signals_path, meta_path, path_col="record_path"):
    signals_path, meta_path = Path(signals_path), Path(meta_path)
    signals_path.parent.mkdir(parents=True, exist_ok=True)
    signals = np.lib.format.open_memmap(str(signals_path), mode="w+", dtype=np.float32, shape=(len(df), 2800, 12))
    rows = []
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="preprocess")):
        signals[i] = preprocess_record(row[path_col])
        d = row.to_dict(); d["signal_idx"] = i; rows.append(d)
    signals.flush()
    out = pd.DataFrame(rows)
    out.to_csv(meta_path, index=False)
    return out


def compute_train_stats(signals, train_indices):
    sum_c = np.zeros(signals.shape[2], dtype=np.float64)
    sumsq_c = np.zeros(signals.shape[2], dtype=np.float64)
    n = 0
    for idx in tqdm(train_indices, desc="train stats"):
        s = signals[int(idx)].astype(np.float32)
        sum_c += s.sum(0); sumsq_c += (s**2).sum(0); n += s.shape[0]
    mean_c = (sum_c / n).astype(np.float32)
    std_c = np.sqrt(np.maximum(sumsq_c / n - mean_c**2, 0)).astype(np.float32)
    return mean_c, std_c


def build_normalized(train_df, val_df, test_df, src_signals_path, dst_signals_path, dst_meta_path,
                     stats_path=None, augment_fn=None,
                     n_aug_samitrop=10, n_aug_code15=5,
                     label_col="chagas_label", samitrop_group="samitrop",
                     src_idx_col="signal_idx", seed=42):
    src = np.load(str(src_signals_path), mmap_mode="r")
    T, C = src.shape[1], src.shape[2]
    Path(dst_signals_path).parent.mkdir(parents=True, exist_ok=True)

    mean_c, std_c = compute_train_stats(src, train_df[src_idx_col].astype(int).to_numpy())
    if stats_path:
        np.savez(str(stats_path), mean=mean_c, std=std_c)

    def n_aug(row):
        if augment_fn is None or row[label_col] != 1:
            return 0
        return n_aug_samitrop if row["dataset_group"] == samitrop_group else n_aug_code15

    n_total = len(train_df) + sum(n_aug(r) for _, r in train_df.iterrows()) + len(val_df) + len(test_df)
    dst = np.lib.format.open_memmap(str(dst_signals_path), mode="w+", dtype=np.float32, shape=(n_total, T, C))

    rng, rows, wi = np.random.default_rng(seed), [], 0

    def write(signal, row_dict, split, is_aug):
        nonlocal wi
        dst[wi] = ((signal - mean_c) / (std_c + 1e-6)).astype(np.float32)
        rows.append({**row_dict, "norm_idx": wi, "split": split, "is_augmented": is_aug})
        wi += 1

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="train"):
        orig = src[int(row[src_idx_col])].astype(np.float32)
        write(orig, row.to_dict(), "train", False)
        for _ in range(n_aug(row)):
            np.random.seed(int(rng.integers(0, 2**31 - 1)))
            write(augment_fn(orig.copy()), row.to_dict(), "train", True)

    for split, split_df in [("val", val_df), ("test", test_df)]:
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split):
            write(src[int(row[src_idx_col])].astype(np.float32), row.to_dict(), split, False)

    dst.flush()
    out = pd.DataFrame(rows[:wi])
    out.to_csv(dst_meta_path, index=False)
    return out, mean_c, std_c


def make_archive(src_path, archive_path):
    src_path, archive_path = Path(src_path), Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w") as t:
        t.add(src_path, arcname=src_path.name)
    return archive_path.stat().st_size
