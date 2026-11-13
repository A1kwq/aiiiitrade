"""
Backtest engine for crypto trading strategies.

Provides a backtesting framework that simulates trading on historical data.
"""
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime


class BacktestEngine:
    """Backtesting engine for evaluating trading strategies."""

    def __init__(self, initial_capital: float = 10_000.0):
        self.initial_capital = initial_capital
        self.results: List[Dict[str, Any]] = []

    def run(self, data: np.ndarray, strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a backtest on historical data with given strategy parameters.

        Args:
            data: numpy array with columns [timestamp, open, high, low, close, volume]
            strategy_params: dict with keys like 'rsi_period', 'rsi_overbought',
                'rsi_oversold', 'macd_signal', 'macd_threshold', 'risk_per_trade'

        Returns:
            dict with keys: 'total_return', 'num_trades', 'win_rate', 'max_drawdown',
                'sharpe_ratio', 'equity_curve'
        """
        if len(data) < 30:
            return {
                'total_return': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'equity_curve': [self.initial_capital],
            }

        close = data[:, 4]
        high = data[:, 2]
        low = data[:, 3]
        open_ = data[:, 1]

        # Compute indicators
        rsi_period = strategy_params.get('rsi_period', 14)
        rsi_overbought = strategy_params.get('rsi_overbought', 70)
        rsi_oversold = strategy_params.get('rsi_oversold', 30)
        macd_signal = strategy_params.get('macd_signal', 12)
        macd_threshold = strategy_params.get('macd_threshold', 0.001)
        risk_per_trade = strategy_params.get('risk_per_trade', 0.02)

        rsi = self._compute_rsi(close, rsi_period)

        # MACD
        ema_fast = self._ema(close, macd_signal)
        ema_slow = self._ema(close, macd_signal * 2)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, 9)

        equity = [self.initial_capital]
        trades: List[Dict] = []
        position = 0.0  # 0 = no position, 1 = long
        entry_price = 0.0

        for i in range(1, len(close)):
            if position == 0:
                # Entry conditions: RSI oversold AND MACD bullish crossover
                rsi_signal = rsi[i] < rsi_oversold
                macd_cross = (macd_line[i] > signal_line[i]) and (macd_line[i-1] <= signal_line[i-1])
                if rsi_signal and macd_cross:
                    position = 1
                    entry_price = close[i]
            elif position == 1:
                # Exit conditions: RSI overbought OR MACD bearish crossover
                rsi_exit = rsi[i] > rsi_overbought
                macd_exit = (macd_line[i] < signal_line[i]) and (macd_line[i-1] >= signal_line[i-1])
                if rsi_exit or macd_exit:
                    pnl = (close[i] - entry_price) / entry_price * 100
                    position = 0
                    trades.append({'entry': entry_price, 'exit': close[i], 'pnl_pct': pnl})
            equity.append(self.initial_capital * (1 + sum(t['pnl_pct'] for t in trades) / 100))

        total_return = (equity[-1] - self.initial_capital) / self.initial_capital * 100
        num_trades = len(trades)
        wins = sum(1 for t in trades if t['pnl_pct'] > 0)
        win_rate = wins / num_trades * 100 if num_trades > 0 else 0.0

        # Max drawdown
        max_drawdown = 0.0
        peak = equity[0]
        for val in equity:
            peak = max(peak, val)
            dd = (peak - val) / peak * 100
            max_drawdown = max(max_drawdown, dd)

        # Sharpe ratio (simplified)
        returns = np.diff(equity) / np.array(equity[:-1])
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        return {
            'total_return': round(total_return, 2),
            'num_trades': num_trades,
            'win_rate': round(win_rate, 1),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe, 2),
            'equity_curve': equity,
            'trades': trades,
        }

    def _compute_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Compute RSI indicator."""
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)

        # Simple moving average for RSI (simplified)
        for i in range(len(prices)):
            if i == 0:
                avg_gain[0] = gain[0]
                avg_loss[0] = loss[0]
            else:
                avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period

        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 1.0)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _ema(self, values: np.ndarray, period: int) -> np.ndarray:
        """Compute Exponential Moving Average."""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(values)
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
        return ema
