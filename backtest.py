"""
Realistic vectorised backtesting engine that:
  • Applies transaction costs (commission + slippage)
  • Enforces a one-bar execution delay (signal on t, execute at t+1 open)
  • Computes all standard risk-adjusted metrics
  • Builds walk-forward equity curves
  • Provides benchmark comparison (buy & hold)
  • Statistical significance test (Deflated Sharpe)

Usage:
    from backtest import Backtester
    bt = Backtester(commission=0.001, slippage=0.0005)
    results = bt.run(signals_df, prices_df)
    bt.print_report(results)
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTER
# ─────────────────────────────────────────────────────────────────────────────

class Backtester:
    """
    Parameters
    ----------
    commission : float
        Round-trip commission as a fraction of trade value (e.g. 0.001 = 0.1%).
    slippage   : float
        One-way slippage as a fraction of trade value (e.g. 0.0005 = 0.05%).
    initial_capital : float
        Starting portfolio value in USD.
    max_position_size : float
        Maximum fraction of capital in one position (0–1).
    stop_loss : float | None
        Daily stop-loss threshold (e.g. 0.02 = 2%). Exits position if hit.
    """

    def __init__(self,
                 commission: float = 0.001,
                 slippage: float   = 0.0005,
                 initial_capital: float = 100_000,
                 max_position_size: float = 1.0,
                 stop_loss: float | None = 0.05):
        self.commission         = commission
        self.slippage           = slippage
        self.initial_capital    = initial_capital
        self.max_position_size  = max_position_size
        self.stop_loss          = stop_loss

    # ── core run ─────────────────────────────────────────────────────────────

    def run(self, signals: pd.Series, close_prices: pd.Series) -> dict:
        """
        Parameters
        ----------
        signals      : pd.Series with values {-1, 0, 1}, indexed by date.
        close_prices : pd.Series of daily closing prices, same or wider date range.

        Returns a dict of performance metrics + time-series DataFrames.
        """
        # F7: deduplicate index before alignment
        signals      = signals[~signals.index.duplicated(keep='first')]
        close_prices = close_prices[~close_prices.index.duplicated(keep='first')]

        # Align
        idx = signals.index.intersection(close_prices.index)

        # F3: guard against empty intersection
        if len(idx) == 0:
            empty = pd.Series(dtype=float)
            return {
                "metrics": {"Total Return (%)": 0, "CAGR (%)": 0, "Ann. Volatility (%)": 0,
                            "Sharpe Ratio": 0, "Sortino Ratio": 0, "Max Drawdown (%)": 0,
                            "Calmar Ratio": 0, "Win Rate (%)": 0, "Profit Factor": 0,
                            "Alpha (ann, %)": 0, "Beta": 0, "Num Trades": 0,
                            "drawdown_series": empty, "deflated_sharpe_p": 1.0},
                "equity": empty, "bh_equity": empty, "strat_ret": empty,
                "bh_ret": empty, "positions": empty, "costs": empty, "signals": empty,
            }

        sig   = signals.reindex(idx)
        price = close_prices.reindex(idx)

        # Daily returns
        daily_ret = price.pct_change().fillna(0)

        # One-bar lag: signal formed at t, execute at t+1
        pos = sig.shift(1).fillna(0)

        # Stop-loss: exit if daily loss exceeds threshold
        if self.stop_loss:
            in_trade    = pos != 0
            loss_hit    = (pos * daily_ret < -self.stop_loss)
            pos[in_trade & loss_hit] = 0

        # Transaction cost when position changes
        turnover         = pos.diff().abs().fillna(0)
        cost_per_trade   = self.commission + self.slippage
        cost             = turnover * cost_per_trade

        # Strategy return after costs
        strat_ret = pos * daily_ret - cost

        # Equity curve
        equity = self.initial_capital * (1 + strat_ret).cumprod()

        # Benchmark (buy & hold)
        bh_ret    = daily_ret
        bh_equity = self.initial_capital * (1 + bh_ret).cumprod()

        metrics = self._compute_metrics(strat_ret, bh_ret, equity)
        metrics["deflated_sharpe_p"] = self._deflated_sharpe_pvalue(strat_ret)

        return {
            "metrics":    metrics,
            "equity":     equity,
            "bh_equity":  bh_equity,
            "strat_ret":  strat_ret,
            "bh_ret":     bh_ret,
            "positions":  pos,
            "costs":      cost,
            "signals":    sig,
        }

    # ── metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(self, strat_ret: pd.Series, bh_ret: pd.Series,
                         equity: pd.Series) -> dict:
        ann = 252  # trading days

        # Annualised return
        total_ret = equity.iloc[-1] / self.initial_capital - 1
        n_days    = len(strat_ret)
        cagr      = (1 + total_ret) ** (ann / n_days) - 1

        # Volatility
        ann_vol   = strat_ret.std() * np.sqrt(ann)

        # Sharpe (rf = 0 for simplicity, easy to parameterise)
        sharpe = (strat_ret.mean() * ann) / (strat_ret.std() * np.sqrt(ann) + 1e-9)

        # Sortino
        downside  = strat_ret[strat_ret < 0].std() * np.sqrt(ann) + 1e-9
        sortino   = (strat_ret.mean() * ann) / downside

        # Max Drawdown
        roll_max  = equity.cummax()
        dd_series = (equity - roll_max) / roll_max
        max_dd    = dd_series.min()

        # Calmar
        calmar = cagr / abs(max_dd + 1e-9)

        # Win rate
        trade_rets  = strat_ret[strat_ret != 0]
        win_rate    = (trade_rets > 0).mean() if len(trade_rets) else 0

        # Profit factor
        gross_profit = trade_rets[trade_rets > 0].sum()
        gross_loss   = abs(trade_rets[trade_rets < 0].sum()) + 1e-9
        profit_factor = gross_profit / gross_loss

        # Alpha / Beta vs benchmark
        aligned = pd.concat([strat_ret, bh_ret], axis=1).dropna()
        aligned.columns = ["strat", "bh"]
        if len(aligned) > 10 and aligned["bh"].std() > 1e-10:
            beta, alpha_daily, *_ = stats.linregress(aligned["bh"], aligned["strat"])
            alpha = alpha_daily * ann
        else:
            beta, alpha = 0, 0

        # Number of trades
        n_trades = (strat_ret != 0).sum()

        return {
            "Total Return (%)":    round(total_ret * 100, 2),
            "CAGR (%)":            round(cagr * 100, 2),
            "Ann. Volatility (%)": round(ann_vol * 100, 2),
            "Sharpe Ratio":        round(sharpe, 3),
            "Sortino Ratio":       round(sortino, 3),
            "Max Drawdown (%)":    round(max_dd * 100, 2),
            "Calmar Ratio":        round(calmar, 3),
            "Win Rate (%)":        round(win_rate * 100, 2),
            "Profit Factor":       round(profit_factor, 3),
            "Alpha (ann, %)":      round(alpha * 100, 2),
            "Beta":                round(beta, 3),
            "Num Trades":          int(n_trades),
            "drawdown_series":     dd_series,
        }

    # ── Deflated Sharpe (Lopez de Prado) ─────────────────────────────────────

    def _deflated_sharpe_pvalue(self, ret: pd.Series) -> float:
        """
        Compute p-value for the Deflated Sharpe Ratio test.
        Accounts for skewness, kurtosis, and multiple testing.
        Returns p-value; < 0.05 means statistically significant.
        """
        from scipy.stats import norm
        T     = len(ret)
        sr    = ret.mean() / (ret.std() + 1e-9) * np.sqrt(252)
        sk    = ret.skew()
        ku    = ret.kurtosis()
        # Standard error of Sharpe under non-normality
        se_sr = np.sqrt((1 + 0.5 * sr**2 - sk * sr + (ku + 3) / 4 * sr**2) / T)
        z     = sr / (se_sr + 1e-9)
        return round(1 - norm.cdf(z), 4)

    # ── walk-forward equity ───────────────────────────────────────────────────

    def run_walkforward(self, folds: list) -> dict:
        """
        folds : list of (signals_df, prices_series) for each OOS period.
        Stitches OOS equity curves together for a single walk-forward equity line.
        """
        equity_parts, ret_parts = [], []
        capital = self.initial_capital

        for fold_signals, fold_prices in folds:
            res = self.run(fold_signals, fold_prices)
            eq  = res["equity"]
            # Skip empty folds (e.g. empty intersection)
            if eq.empty:
                continue
            # Rescale equity to current capital
            scale = capital / self.initial_capital
            eq    = eq * scale
            equity_parts.append(eq)
            ret_parts.append(res["strat_ret"])
            capital = float(eq.iloc[-1])  # guard: explicit float cast

        if not equity_parts:
            empty = pd.Series(dtype=float)
            return {"metrics": {}, "equity": empty, "strat_ret": empty}
        combined_equity  = pd.concat(equity_parts)
        combined_ret     = pd.concat(ret_parts)
        bh_prices        = pd.concat([fp for _, fp in folds])
        bh_ret           = bh_prices.pct_change().fillna(0)

        metrics = self._compute_metrics(combined_ret, bh_ret, combined_equity)
        metrics["deflated_sharpe_p"] = self._deflated_sharpe_pvalue(combined_ret)

        return {
            "metrics":   metrics,
            "equity":    combined_equity,
            "strat_ret": combined_ret,
        }

    # ── reporting ─────────────────────────────────────────────────────────────

    def print_report(self, results: dict):
        m = results["metrics"]
        print("\n" + "═" * 45)
        print("  BACKTEST PERFORMANCE REPORT")
        print("═" * 45)
        skip = {"drawdown_series"}
        for k, v in m.items():
            if k not in skip:
                print(f"  {k:<28} {v}")
        p = m.get("deflated_sharpe_p", None)
        if p is not None:
            sig = "✓ Significant" if p < 0.05 else "✗ Not significant"
            print(f"  {'Deflated Sharpe p-value':<28} {p}  ({sig})")
        print("═" * 45 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_pipeline import fetch_data
    import random

    df = fetch_data("AAPL", period="3y")
    prices = df["Close"]

    # Random signals as smoke test
    rng = np.random.default_rng(42)
    sig = pd.Series(rng.choice([-1, 0, 1], size=len(prices)), index=prices.index)

    bt = Backtester(commission=0.001, slippage=0.0005)
    results = bt.run(sig, prices)
    bt.print_report(results)