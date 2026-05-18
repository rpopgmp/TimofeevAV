import numpy as np
import pandas as pd


LEAKAGE_COLS = {
    "signal_idx", "norm_idx", "record_path", "record_name",
    "source", "dataset_group", "lead_names", "chagas_label",
    "split", "is_augmented", "sex", "error",
}


def select_feature_cols(df, exclude=None):
    excl = set(exclude or []) | LEAKAGE_COLS
    return [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]


def make_splits(df, samitrop_group="samitrop", n_test_samitrop=500,
                neg_test_frac=0.20, val_frac=0.25,
                strength_col=None, label_col="chagas_label", random_state=42):
    df = df.reset_index(drop=True).copy()
    rng = np.random.RandomState(random_state)

    is_sami, is_pos, is_neg = df["dataset_group"] == samitrop_group, df[label_col] == 1, df[label_col] == 0

    sami_pos = df[is_sami & is_pos]
    if strength_col and strength_col in sami_pos.columns:
        test_sami = sami_pos.sort_values(strength_col, ascending=False).head(n_test_samitrop).index.to_numpy()
    else:
        test_sami = rng.permutation(sami_pos.index.to_numpy())[:n_test_samitrop]

    neg_perm = rng.permutation(df[is_neg].index.to_numpy())
    n_test_neg = int(round(neg_test_frac * len(neg_perm)))
    remaining_neg = neg_perm[n_test_neg:]
    n_val_neg = int(round(val_frac * len(remaining_neg)))

    pos_rem = rng.permutation(df[is_pos & ~df.index.isin(test_sami)].index.to_numpy())
    n_val_pos = int(round(val_frac * len(pos_rem)))

    train_idx = np.concatenate([pos_rem[n_val_pos:], remaining_neg[n_val_neg:]])
    val_idx   = np.concatenate([pos_rem[:n_val_pos], remaining_neg[:n_val_neg]])
    test_idx  = np.concatenate([test_sami, neg_perm[:n_test_neg]])

    train_df = df.loc[train_idx].assign(split="train").reset_index(drop=True)
    val_df   = df.loc[val_idx  ].assign(split="val"  ).reset_index(drop=True)
    test_df  = df.loc[test_idx ].assign(split="test" ).reset_index(drop=True)

    total = len(train_df) + len(val_df) + len(test_df)
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name}: {len(d)} ({100*len(d)/total:.0f}%) | "
              f"pos={int((d[label_col]==1).sum())} neg={int((d[label_col]==0).sum())}")
    return train_df, val_df, test_df


def vif_select(df, feature_cols, threshold=10.0, verbose=True):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(df[feature_cols].median(numeric_only=True))
    cols = list(feature_cols)
    while True:
        Xc = add_constant(X[cols], has_constant="add")
        vifs = [(col, variance_inflation_factor(Xc.values, i + 1)) for i, col in enumerate(cols)]
        worst_col, worst_v = max(vifs, key=lambda x: x[1] if np.isfinite(x[1]) else x[1])
        if not np.isfinite(worst_v) or worst_v > threshold:
            if verbose:
                print(f"drop {worst_col} (VIF={worst_v:.2f}) | remaining={len(cols)-1}")
            cols.remove(worst_col)
            if not cols:
                break
        else:
            break
    return cols
