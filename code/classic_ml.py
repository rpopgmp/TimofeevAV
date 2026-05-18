import joblib
from pathlib import Path

import numpy as np
import lightgbm as lgb
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def _xy(df, cols, label_col="chagas_label"):
    X = np.nan_to_num(df[cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -1e6, 1e6), df[label_col].to_numpy(dtype=np.int32)


def _save(path, obj):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path)


def train_logreg(train_df, val_df, feature_cols, label_col="chagas_label", save_path=None):
    X_tr, y_tr = _xy(train_df, feature_cols, label_col)
    X_val, y_val = _xy(val_df, feature_cols, label_col)
    scaler = StandardScaler().fit(X_tr)
    model = LogisticRegression(penalty=None, max_iter=1000, class_weight="balanced")
    model.fit(scaler.transform(X_tr), y_tr)
    print(f"logreg val AUC: {roc_auc_score(y_val, model.predict_proba(scaler.transform(X_val))[:, 1]):.4f}")
    _save(save_path, {"model": model, "scaler": scaler, "feature_cols": list(feature_cols)})
    return model, scaler


def predict_logreg(model, scaler, df, feature_cols):
    X, _ = _xy(df, feature_cols)
    return model.predict_proba(scaler.transform(X))[:, 1]


def train_knn_optuna(train_df, val_df, feature_cols, label_col="chagas_label", n_trials=20, save_path=None):
    X_tr, y_tr = _xy(train_df, feature_cols, label_col)
    X_val, y_val = _xy(val_df, feature_cols, label_col)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_val_s = scaler.transform(X_tr), scaler.transform(X_val)

    def objective(trial):
        m = KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors", 3, 51, step=2),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            metric=trial.suggest_categorical("metric", ["euclidean", "manhattan", "cosine"]),
            n_jobs=-1,
        )
        m.fit(X_tr_s, y_tr)
        return roc_auc_score(y_val, m.predict_proba(X_val_s)[:, 1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    model = KNeighborsClassifier(**study.best_params, n_jobs=-1)
    model.fit(X_tr_s, y_tr)
    print(f"knn val AUC: {roc_auc_score(y_val, model.predict_proba(X_val_s)[:, 1]):.4f}")
    _save(save_path, {"model": model, "scaler": scaler, "feature_cols": list(feature_cols)})
    return model, scaler, study


def predict_knn(model, scaler, df, feature_cols):
    X, _ = _xy(df, feature_cols)
    return model.predict_proba(scaler.transform(X))[:, 1]


def train_svm_optuna(train_df, val_df, feature_cols, label_col="chagas_label", n_trials=20, save_path=None):
    X_tr, y_tr = _xy(train_df, feature_cols, label_col)
    X_val, y_val = _xy(val_df, feature_cols, label_col)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_val_s = scaler.transform(X_tr), scaler.transform(X_val)

    def objective(trial):
        m = LinearSVC(
            C=trial.suggest_float("C", 1e-3, 1e2, log=True),
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
            dual="auto",
        )
        m.fit(X_tr_s, y_tr)
        return roc_auc_score(y_val, m.decision_function(X_val_s))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    model = LinearSVC(
        **study.best_params,
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
        dual="auto",
    )
    model.fit(X_tr_s, y_tr)
    print(f"linsvc val AUC: {roc_auc_score(y_val, model.decision_function(X_val_s)):.4f}")
    _save(save_path, {"model": model, "scaler": scaler, "feature_cols": list(feature_cols)})
    return model, scaler, study


def predict_svm(model, scaler, df, feature_cols):
    X, _ = _xy(df, feature_cols)
    scores = model.decision_function(scaler.transform(X))
    # сигмоида от decision_function -> псевдо-вероятности в [0, 1] для downstream-стекинга
    return 1.0 / (1.0 + np.exp(-scores))


def train_lgbm_optuna(train_df, val_df, feature_cols, label_col="chagas_label",
                     n_trials=15, random_state=42, save_path=None):
    X_tr, y_tr = _xy(train_df, feature_cols, label_col)
    X_val, y_val = _xy(val_df, feature_cols, label_col)
    scale_pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    def objective(trial):
        m = lgb.LGBMClassifier(
            objective="binary", metric="auc", verbose=-1,
            scale_pos_weight=scale_pos_weight, random_state=random_state,
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    model = lgb.LGBMClassifier(
        **study.best_params, objective="binary", metric="auc", verbose=-1,
        scale_pos_weight=scale_pos_weight, random_state=random_state,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    print(f"lgbm val AUC: {roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]):.4f}")
    _save(save_path, {"model": model, "feature_cols": list(feature_cols)})
    return model, study


def predict_lgbm(model, df, feature_cols):
    X, _ = _xy(df, feature_cols)
    return model.predict_proba(X)[:, 1]
