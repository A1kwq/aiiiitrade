
"""AI Trader - Trainable model that works on charts"""
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional
import json
import os


@dataclass
class Signal:
    timestamp: str
    symbol: str
    direction: str  # "buy", "sell", "hold"
    confidence: float
    price: float
    reason: str


class AITrader:
    """Trainable AI that analyzes candlestick charts and generates trading signals."""

    def __init__(self):
        self.weights = np.random.randn(128, 64).astype(np.float32) * 0.1
        self.bias1 = np.zeros(64).astype(np.float32)
        self.weights2 = np.random.randn(64, 32).astype(np.float32) * 0.1
        self.bias2 = np.zeros(32).astype(np.float32)
        self.weights3 = np.random.randn(32, 1).astype(np.float32) * 0.1
        self.bias3 = np.zeros(1).astype(np.float32)
        self.learning_rate = 0.001
        self.epoch_count = 0
        self.recent_signals: List[Signal] = []
        self._load_model()

    def _load_model(self):
        """Load trained model if it exists."""
        if os.path.exists("models/ai_trader_weights.json"):
            with open("models/ai_trader_weights.json") as f:
                d = json.load(f)
                self.weights = np.array(d["w1"], dtype=np.float32)
                self.bias1 = np.array(d["b1"], dtype=np.float32)
                self.weights2 = np.array(d["w2"], dtype=np.float32)
                self.bias2 = np.array(d["b2"], dtype=np.float32)
                self.weights3 = np.array(d["w3"], dtype=np.float32)
                self.bias3 = np.array(d["b3"], dtype=np.float32)
                self.epoch_count = d.get("epochs", 0)

    def _save_model(self):
        """Save trained model."""
        os.makedirs("models", exist_ok=True)
        with open("models/ai_trader_weights.json", "w") as f:
            json.dump({
                "w1": self.weights.tolist(),
                "b1": self.bias1.tolist(),
                "w2": self.weights2.tolist(),
                "b2": self.bias2.tolist(),
                "w3": self.weights3.tolist(),
                "b3": self.bias3.tolist(),
                "epochs": self.epoch_count,
            }, f)

    def _forward(self, x):
        h1 = np.tanh(x @ self.weights + self.bias1)
        h2 = np.tanh(h1 @ self.weights2 + self.bias2)
        out = 1 / (1 + np.exp(-((h2 @ self.weights3) + self.bias3)))
        return out, h1, h2

    def _candles_to_features(self, candles: List[dict]) -> np.ndarray:
        """Convert candle data to feature vector for the model."""
        closes = np.array([c["close"] for c in candles], dtype=np.float32)
        opens = np.array([c["open"] for c in candles], dtype=np.float32)
        highs = np.array([c["high"] for c in candles], dtype=np.float32)
        lows = np.array([c["low"] for c in candles], dtype=np.float32)

        # Technical indicators as features
        n = len(closes)
        if n < 14:
            return np.zeros(128, dtype=np.float32)

        # Returns
        rets = np.diff(closes) / closes[:-1]
        # RSI-like
        up = np.where(rets > 0, rets, 0)
        dn = np.where(rets < 0, -rets, 0)
        avg_up = up[-14:].mean()
        avg_dn = dn[-14:].mean()
        rs = avg_up / (avg_dn + 1e-8)

        # Volume trend
        vols = np.array([c.get("volume", 0) for c in candles], dtype=np.float32)
        vol_ratio = vols[-5:].mean() / (vols.mean() + 1e-8) if n > 5 else 1.0

        # Bollinger-like band width
        std14 = closes[-14:].std()
        mean14 = closes[-14:].mean()
        bb_width = (2 * std14) / (mean14 + 1e-8) if mean14 > 0 else 0

        # Price momentum
        mom_7 = (closes[-1] - closes[-8]) / (closes[-8] + 1e-8) if n >= 8 else 0
        mom_30 = (closes[-1] - closes[-30]) / (closes[-30] + 1e-8) if n >= 30 else 0

        # Build feature vector of size 128
        feats = np.zeros(128, dtype=np.float32)
        feats[0] = rs
        feats[1] = vol_ratio
        feats[2] = bb_width
        feats[3] = mom_7
        feats[4] = mom_30
        feats[5] = rets[-1] if len(rets) > 0 else 0
        feats[6] = (highs[-1] - lows[-1]) / (closes[-1] + 1e-8)
        feats[7] = closes[-1] / mean14 - 1
        return feats

    def predict(self, symbol: str, candles: List[dict]) -> Signal:
        """Generate a trading signal from chart data."""
        feats = self._candles_to_features(candles)
        out, _, _ = self._forward(feats.reshape(1, -1))
        prob = float(out[0][0])

        if prob > 0.6:
            direction = "buy"
        elif prob < 0.4:
            direction = "sell"
        else:
            direction = "hold"

        price = candles[-1]["close"] if candles else 0
        sig = Signal(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            direction=direction,
            confidence=abs(prob - 0.5) * 2,
            price=price,
            reason=f"Model output={prob:.3f}"
        )
        self.recent_signals.append(sig)
        if len(self.recent_signals) > 100:
            self.recent_signals = self.recent_signals[-100:]
        return sig

    def train(self, data: dict) -> dict:
        """Train the model on provided candle data."""
        symbol = data.get("symbol", "BTCUSDT")
        candles = data.get("candles", [])
        epochs = data.get("epochs", 50)

        if len(candles) < 30:
            return {"status": "error", "message": "Need at least 30 candles"}

        # Generate training samples from candle history
        X, y = self._make_training_set(candles)
        if len(X) == 0:
            return {"status": "error", "message": "Not enough data to train"}

        for ep in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                out, h1, h2 = self._forward(X[i].reshape(1, -1))
                pred = out[0][0]
                target = y[i]
                loss = (pred - target) ** 2
                total_loss += loss

            # Simple gradient step (approximate for demo)
            lr = self.learning_rate
            for i in range(len(X)):
                out, h1, h2 = self._forward(X[i].reshape(1, -1))
                err = (out[0][0] - y[i])
                # Update output layer
                self.weights3 += lr * np.outer(h2[0], err)
                self.bias3 += lr * err

            self.epoch_count += 1

        self._save_model()
        return {
            "status": "ok",
            "epochs_run": epochs,
            "total_epochs": self.epoch_count,
            "loss": float(total_loss / len(X)),
            "samples": len(X),
        }

    def _make_training_set(self, candles: List[dict]):
        """Create (features, target) pairs from candle history."""
        X, y = [], []
        window = 30
        for i in range(window, len(candles)):
            chunk = candles[i - window:i]
            feats = self._candles_to_features(chunk)
            # Target: next candle direction (1=up, 0=down)
            next_ret = (candles[i]["close"] - candles[i - 1]["close"]) / candles[i - 1]["close"]
            target = 1.0 if next_ret > 0 else 0.0
            X.append(feats)
            y.append(target)
        return np.array(X), np.array(y)

    # ── Risk Management ─────────────────────────────────────────────

    def risk_plan(self, symbol: str, direction: str, confidence: float,
                  entry_price: float, account_balance: float = 1000.0) -> dict:
        """Calculate stop-loss, take-profit and position size."""
        sl_pct = 0.02 * (1 - confidence * 0.5)   # 2% base, reduced with confidence
        tp_pct = 0.04 * (1 + confidence * 0.5)   # 4% base, increased with confidence
        risk_per_trade = account_balance * 0.02  # 2% of balance per trade
        position_value = risk_per_trade / sl_pct
        stop_loss = entry_price * (1 - sl_pct) if direction == "buy" else entry_price * (1 + sl_pct)
        take_profit = entry_price * (1 + tp_pct) if direction == "buy" else entry_price * (1 - tp_pct)
        return {
            "direction": direction,
            "entry": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "position_value": round(position_value, 2),
            "risk_pct": round(sl_pct * 100, 1),
            "reward_pct": round(tp_pct * 100, 1),
            "risk_reward_ratio": round(tp_pct / sl_pct, 2),
        }

    def market_sentiment(self) -> dict:
        """Analyze overall market sentiment across tracked symbols."""
        result = {"symbols": {}, "overall": "neutral", "score": 0.5}
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            sig = self.predict(sym)
            result["symbols"][sym] = {"direction": sig.direction, "confidence": round(sig.confidence, 3)}
        avg_conf = sum(v["confidence"] for v in result["symbols"].values()) / len(result["symbols"])
        buys = sum(1 for v in result["symbols"].values() if v["direction"] == "buy")
        result["overall"] = "bullish" if buys >= 2 else ("bearish" if buys == 0 else "neutral")
        result["score"] = round(avg_conf, 3)
        return result

    def backtest(self, symbol: str, candles: List[dict], lookback: int = 30) -> dict:
        """Run a simple backtest over candle history."""
        if len(candles) < lookback + 5:
            return {"status": "error", "message": "Not enough candles"}
        wins, losses, total_pnl = 0, 0, 0.0
        trades = []
        for i in range(lookback, len(candles)):
            chunk = candles[i - lookback:i]
            feats = self._candles_to_features(chunk)
            out, _, _ = self._forward(feats.reshape(1, -1))
            prob = float(out[0][0])
            direction = "buy" if prob > 0.5 else "sell"
            entry = candles[i - 1]["close"]
            exit_p = candles[i]["close"]
            pnl = (exit_p - entry) / entry * 100 if direction == "buy" else (entry - exit_p) / entry * 100
            total_pnl += pnl
            if pnl > 0:
                wins += 1
            else:
                losses += 1
            trades.append({"idx": i, "direction": direction, "pnl": round(pnl, 4)})
        win_rate = round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0
        return {
            "status": "ok",
            "symbol": symbol,
            "total_trades": wins + losses,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl_pct": round(total_pnl, 3),
            "avg_pnl": round(total_pnl / max(wins + losses, 1), 4),
        }

    def get_candles(self, symbol: str, tf: str) -> List[dict]:
        """Get candle data - in production this hits an exchange API."""
        # For now, generate realistic sample data
        return self._generate_sample_candles(symbol, tf)

    def _generate_sample_candles(self, symbol: str, tf: str) -> List[dict]:
        """Generate realistic OHLCV candles for chart display."""
        np.random.seed(42)
        n = 100
        base = 67500.0
        prices = []
        for _ in range(n):
            change = np.random.randn() * 0.008
            p = base * (1 + sum(np.random.randn(50) * 0.003))
            prices.append(p)

        candles = []
        for i, p in enumerate(prices):
            o = p * (1 + np.random.randn() * 0.002)
            c = p * (1 + np.random.randn() * 0.002)
            h = max(o, c) * (1 + abs(np.random.randn()) * 0.003)
            l = min(o, c) * (1 - abs(np.random.randn()) * 0.003)
            v = abs(np.random.randn() * 1000 + 500)
            ts = f"2024-01-{(i // 24) + 1:02d}"
            candles.append({
                "time": ts,
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "volume": round(v, 1)
            })
        return candles

    def get_stats(self) -> dict:
        buys = sum(1 for s in self.recent_signals if s.direction == "buy")
        sells = sum(1 for s in self.recent_signals if s.direction == "sell")
        holds = sum(1 for s in self.recent_signals if s.direction == "hold")
        return {
            "total_signals": len(self.recent_signals),
            "buys": buys,
            "sells": sells,
            "holds": holds,
            "model_epochs": self.epoch_count,
            "last_signal": self.recent_signals[-1].direction if self.recent_signals else "none",
        }

    def get_signals(self) -> List[dict]:
        return [
            {
                "timestamp": s.timestamp,
                "symbol": s.symbol,
                "direction": s.direction,
                "confidence": round(s.confidence, 3),
                "price": s.price,
                "reason": s.reason,
            } for s in self.recent_signals[-20:]
        ]
