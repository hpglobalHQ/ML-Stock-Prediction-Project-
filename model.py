"""
Three-layer ensemble:
  Layer 1  →  LSTM  (captures temporal patterns)
  Layer 1  →  XGBoost (captures feature interactions & non-linearities)
  Layer 2  →  Ridge meta-learner (stacks L1 predictions for final output)

Both regression (return magnitude) and classification (direction) heads are
trained simultaneously. Final trading signal combines both.

Usage:
    from model import EnsemblePredictor
    model = EnsemblePredictor(horizon=5)
    model.fit(train_df, val_df, feature_cols)
    signals = model.predict(test_df, feature_cols)
"""

import warnings
warnings.filterwarnings("ignore")

import os, json
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb

from data_pipeline import prepare_xy, get_feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LSTM  (temporal sequential model)
# ─────────────────────────────────────────────────────────────────────────────

class LSTMNet(nn.Module):
    """Bidirectional LSTM with dropout and two output heads."""

    def __init__(self, in_dim: int, hidden: int = 128, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0,
            bidirectional=True,
        )
        self.norm  = nn.LayerNorm(hidden * 2)
        self.drop  = nn.Dropout(dropout)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)           # (B, T, 2*hidden)
        last    = out[:, -1, :]         # last timestep
        last    = self.drop(self.norm(last))
        return self.reg_head(last).squeeze(-1), self.cls_head(last).squeeze(-1)


def make_sequences(X: np.ndarray, y_reg: np.ndarray, y_cls: np.ndarray,
                   seq_len: int = 20):
    """Sliding window → 3D tensor for LSTM."""
    Xs, yr, yc = [], [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len: i])
        yr.append(y_reg[i])
        yc.append(y_cls[i])
    return np.array(Xs, dtype=np.float32), np.array(yr, dtype=np.float32), np.array(yc, dtype=np.float32)


class LSTMPredictor:
    def __init__(self, seq_len: int = 20, hidden: int = 128, epochs: int = 40,
                 lr: float = 1e-3, batch: int = 64, patience: int = 8):
        self.seq_len = seq_len
        self.hidden  = hidden
        self.epochs  = epochs
        self.lr      = lr
        self.batch   = batch
        self.patience = patience
        self.model   = None
        self.scaler  = RobustScaler()
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"

    def _build(self, in_dim):
        self.model = LSTMNet(in_dim, self.hidden).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

    def fit(self, X_tr, y_reg_tr, y_cls_tr, X_val, y_reg_val, y_cls_val):
        X_tr  = self.scaler.fit_transform(X_tr)
        X_val = self.scaler.transform(X_val)

        Xs_tr,  yr_tr,  yc_tr  = make_sequences(X_tr,  y_reg_tr,  y_cls_tr,  self.seq_len)
        Xs_val, yr_val, yc_val = make_sequences(X_val, y_reg_val, y_cls_val, self.seq_len)

        self._build(Xs_tr.shape[-1])

        loader_tr = DataLoader(
            TensorDataset(torch.tensor(Xs_tr), torch.tensor(yr_tr), torch.tensor(yc_tr)),
            batch_size=self.batch, shuffle=False
        )

        bce  = nn.BCELoss()
        mse  = nn.HuberLoss(delta=0.01)  # robust to return outliers
        best_val_loss, no_improve = np.inf, 0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            for Xb, yrb, ycb in loader_tr:
                Xb, yrb, ycb = Xb.to(self.device), yrb.to(self.device), ycb.to(self.device)
                self.optimizer.zero_grad()
                pred_r, pred_c = self.model(Xb)
                loss = mse(pred_r, yrb) + 0.5 * bce(pred_c, ycb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            self.scheduler.step()

            # Validation
            val_loss = self._val_loss(Xs_val, yr_val, yc_val, mse, bce)
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                break

        if best_state:
            self.model.load_state_dict(best_state)
        print(f"[LSTM] Training done. Best val loss: {best_val_loss:.6f}")

    def _val_loss(self, Xs, yr, yc, mse_fn, bce_fn):
        self.model.eval()
        with torch.no_grad():
            Xb = torch.tensor(Xs).to(self.device)
            yr_t = torch.tensor(yr).to(self.device)
            yc_t = torch.tensor(yc).to(self.device)
            pr, pc = self.model(Xb)
            return (mse_fn(pr, yr_t) + 0.5 * bce_fn(pc, yc_t)).item()

    def predict(self, X: np.ndarray, y_reg: np.ndarray, y_cls: np.ndarray):
        """Returns (pred_returns, pred_probs) on the sequential portion of X."""
        X_sc  = self.scaler.transform(X)
        Xs, _, _ = make_sequences(X_sc, y_reg, y_cls, self.seq_len)
        self.model.eval()
        with torch.no_grad():
            Xb = torch.tensor(Xs).to(self.device)
            pr, pc = self.model(Xb)
        return pr.cpu().numpy(), pc.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  XGBoost  (gradient-boosted trees – fast, high accuracy)
# ─────────────────────────────────────────────────────────────────────────────

class XGBPredictor:
    def __init__(self):
        common = dict(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            early_stopping_rounds=30,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        self.reg_model = xgb.XGBRegressor(
            objective="reg:squarederror", eval_metric="rmse", **common
        )
        self.cls_model = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", **common
        )
        self.scaler = RobustScaler()

    def fit(self, X_tr, y_reg_tr, y_cls_tr, X_val, y_reg_val, y_cls_val):
        X_tr  = self.scaler.fit_transform(X_tr)
        X_val = self.scaler.transform(X_val)

        self.reg_model.fit(X_tr, y_reg_tr,
                           eval_set=[(X_val, y_reg_val)],
                           verbose=False)
        self.cls_model.fit(X_tr, y_cls_tr,
                           eval_set=[(X_val, y_cls_val)],
                           verbose=False)
        print("[XGBoost] Training done.")

    def predict(self, X: np.ndarray):
        X_sc = self.scaler.transform(X)
        return self.reg_model.predict(X_sc), self.cls_model.predict_proba(X_sc)[:, 1]

    def feature_importance(self, feature_cols):
        scores = self.reg_model.feature_importances_
        return pd.Series(scores, index=feature_cols).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  META-LEARNER  (stacking)
# ─────────────────────────────────────────────────────────────────────────────

class MetaLearner:
    """Ridge + Logistic stacker that combines LSTM and XGB outputs."""

    def __init__(self):
        self.reg_meta = Ridge(alpha=1.0)
        self.cls_meta = LogisticRegression(C=1.0, max_iter=500, random_state=42)

    def fit(self, lstm_ret, lstm_prob, xgb_ret, xgb_prob, y_reg, y_cls):
        X_meta_reg = np.column_stack([lstm_ret, xgb_ret])
        X_meta_cls = np.column_stack([lstm_prob, xgb_prob])
        self.reg_meta.fit(X_meta_reg, y_reg)
        self.cls_meta.fit(X_meta_cls, y_cls)
        print("[MetaLearner] Stacker trained.")

    def predict(self, lstm_ret, lstm_prob, xgb_ret, xgb_prob):
        X_meta_reg = np.column_stack([lstm_ret, xgb_ret])
        X_meta_cls = np.column_stack([lstm_prob, xgb_prob])
        ret_pred  = self.reg_meta.predict(X_meta_reg)
        prob_pred = self.cls_meta.predict_proba(X_meta_cls)[:, 1]
        return ret_pred, prob_pred


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ENSEMBLE PREDICTOR  (top-level interface)
# ─────────────────────────────────────────────────────────────────────────────

class EnsemblePredictor:
    """
    Full ensemble: LSTM + XGBoost → Ridge/Logistic meta-learner.
    Call .fit() on train/val data, then .predict() for signals.
    """

    def __init__(self, horizon: int = 5, seq_len: int = 20):
        self.horizon  = horizon
        self.seq_len  = seq_len
        self.lstm     = LSTMPredictor(seq_len=seq_len, epochs=40, patience=10)
        self.xgb      = XGBPredictor()
        self.meta     = MetaLearner()
        self.feature_cols = None

    # ── helpers ──────────────────────────────────────────────────────────────

    def _extract(self, df: pd.DataFrame, fit_scaler=False):
        fc   = self.feature_cols
        sub  = df[fc + ["label_ret", "label_dir"]].replace([np.inf, -np.inf], np.nan).dropna()
        X    = sub[fc].values
        y_r  = sub["label_ret"].values
        y_c  = sub["label_dir"].values.astype(int)
        return X, y_r, y_c, sub.index

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list):
        self.feature_cols = feature_cols
        X_tr, yr_tr, yc_tr, _ = self._extract(train_df)
        X_val, yr_val, yc_val, _ = self._extract(val_df)

        # Train base models
        self.lstm.fit(X_tr, yr_tr, yc_tr, X_val, yr_val, yc_val)
        self.xgb.fit(X_tr, yr_tr, yc_tr, X_val, yr_val, yc_val)

        # Generate val predictions for meta-learner training
        lstm_ret_v, lstm_prob_v = self.lstm.predict(X_val, yr_val, yc_val)
        xgb_ret_v, xgb_prob_v  = self.xgb.predict(X_val)

        # Align lengths (LSTM loses seq_len rows)
        offset = len(X_val) - len(lstm_ret_v)
        yr_val_aligned  = yr_val[offset:]
        yc_val_aligned  = yc_val[offset:]
        xgb_ret_v_al    = xgb_ret_v[offset:]
        xgb_prob_v_al   = xgb_prob_v[offset:]

        self.meta.fit(lstm_ret_v, lstm_prob_v, xgb_ret_v_al, xgb_prob_v_al,
                      yr_val_aligned, yc_val_aligned)

        # Eval on val
        final_ret, final_prob = self.meta.predict(
            lstm_ret_v, lstm_prob_v, xgb_ret_v_al, xgb_prob_v_al
        )
        dir_pred = (final_prob > 0.5).astype(int)
        acc = accuracy_score(yc_val_aligned, dir_pred)
        rmse = np.sqrt(mean_squared_error(yr_val_aligned, final_ret))
        print(f"[Ensemble] Val accuracy={acc:.3f}  Val RMSE={rmse:.6f}")
        return {"val_accuracy": acc, "val_rmse": rmse}

    def predict(self, test_df: pd.DataFrame):
        """Returns DataFrame with columns: pred_return, pred_prob, signal"""
        X_te, yr_te, yc_te, idx = self._extract(test_df)

        lstm_ret, lstm_prob = self.lstm.predict(X_te, yr_te, yc_te)
        xgb_ret, xgb_prob  = self.xgb.predict(X_te)

        offset = len(X_te) - len(lstm_ret)
        xgb_ret_al  = xgb_ret[offset:]
        xgb_prob_al = xgb_prob[offset:]
        idx_al      = idx[offset:]

        final_ret, final_prob = self.meta.predict(
            lstm_ret, lstm_prob, xgb_ret_al, xgb_prob_al
        )

        # ── Signal generation ────────────────────────────────────────────────
        # Use XGBoost raw probs as primary signal — better calibrated than
        # the meta-learner when val set is small.
        # final_prob is used only for display (confidence level).
        xgb_p = xgb_prob_al  # well-calibrated [0,1] probs from XGB

        # Semantic signal: 0.5 is the absolute boundary between bullish/bearish.
        # Band width adapts to prob spread but always respects absolute meaning.
        #   prob >= 0.5+band → LONG   (XGB genuinely predicts upward move)
        #   prob <= 0.5-band → SHORT  (XGB genuinely predicts downward move)
        #   otherwise        → HOLD
        p_std = xgb_p.std()
        band  = max(0.02, 0.30 * p_std)   # at least 2% away from 0.5

        hi = 0.5 + band
        lo = 0.5 - band
        signal = np.where(xgb_p >= hi,  1,
                 np.where(xgb_p <= lo, -1, 0))

        # If confidence band is too tight and >80% are HOLD, loosen slightly
        hold_frac = (signal == 0).mean()
        if hold_frac > 0.80:
            tight_band = max(0.005, 0.10 * p_std)
            signal = np.where(xgb_p >= 0.5 + tight_band,  1,
                     np.where(xgb_p <= 0.5 - tight_band, -1, 0))

        # Keep original DatetimeIndex (do NOT convert to date — breaks df.reindex)
        out_idx = idx_al

        return pd.DataFrame({
            "pred_return": final_ret,
            "pred_prob":   xgb_p,   # XGB classification prob (used for signal + display)
            "signal":      signal,
        }, index=out_idx)

    def feature_importance(self):
        return self.xgb.feature_importance(self.feature_cols)

    def save(self, path: str = "artifacts/model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save everything except the torch model (saved separately)
        torch.save(self.lstm.model.state_dict(), path.replace(".pkl", "_lstm.pt"))
        self.lstm.model = None  # avoid pickle issues
        joblib.dump(self, path)
        print(f"[Ensemble] Saved → {path}")

    @classmethod
    def load(cls, path: str = "artifacts/model.pkl"):
        obj = joblib.load(path)
        # Reload torch model
        pt_path = path.replace(".pkl", "_lstm.pt")
        if os.path.exists(pt_path):
            dummy_x = np.zeros((1, obj.lstm.seq_len, len(obj.feature_cols)))
            obj.lstm._build(len(obj.feature_cols))
            obj.lstm.model.load_state_dict(torch.load(pt_path, map_location="cpu"))
            obj.lstm.model.eval()
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_pipeline import build_dataset, walk_forward_splits, get_feature_cols

    df = build_dataset("AAPL", period="5y", horizon=5)
    feature_cols = get_feature_cols(df)

    splits = list(walk_forward_splits(df, n_splits=3))
    train_df, val_df, test_df = splits[0]

    model = EnsemblePredictor(horizon=5, seq_len=20)
    metrics = model.fit(train_df, val_df, feature_cols)
    preds   = model.predict(test_df)
    print(preds.head(10))
    print("Feature importance (top 10):")
    print(model.feature_importance().head(10))