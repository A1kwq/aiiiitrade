
"""Crypto AI Trader - Main Application"""
from flask import Flask, render_template, jsonify, request
import json
import os
from datetime import datetime
import threading

from ai_trader import AITrader
from backtest import BacktestEngine

app = Flask(__name__)
ai_trader = AITrader()
backtester = BacktestEngine()


@app.route("/")
def dashboard():
    return render_template("dashboard.html",
                           stats=ai_trader.get_stats(),
                           recent_signals=ai_trader.recent_signals[:10])


@app.route("/api/signals")
def get_signals():
    return jsonify(ai_trader.get_signals())


@app.route("/api/train", methods=["POST"])
def train_model():
    data = request.json
    result = ai_trader.train(data)
    return jsonify(result)


@app.route("/api/backtest", methods=["POST"])
def run_backtest():
    import numpy as np
    data = request.json
    symbol = data.get("symbol", "BTCUSDT")
    tf = data.get("tf", "4h")
    candles = ai_trader.get_candles(symbol, tf)
    if not candles:
        return jsonify({"status": "error", "message": "No data"})
    arr = np.array([[c["timestamp"], c["open"], c["high"], c["low"], c["close"], c.get("volume", 0)] for c in candles])
    result = backtester.run(arr, {})
    return jsonify(result)


@app.route("/api/chart_data")
def get_chart_data():
    symbol = request.args.get("symbol", "BTCUSDT")
    tf = request.args.get("tf", "4h")
    candles = ai_trader.get_candles(symbol, tf)
    return jsonify(candles)


@app.route("/api/stats")
def get_stats():
    return jsonify(ai_trader.get_stats())


@app.route("/api/positions")
def get_positions():
    """Return active positions (mock data for now)."""
    import numpy as np
    last_sig = ai_trader.recent_signals[-1] if ai_trader.recent_signals else None
    positions = []
    if last_sig:
        direction = last_sig.direction
        if direction in ("buy", "sell"):
            positions.append({
                "symbol": last_sig.symbol,
                "side": "LONG" if direction == "buy" else "SHORT",
                "entry_price": round(last_sig.price, 2),
                "size": 1.0,
                "pnl_pct": round(float(np.random.uniform(-2, 3)), 2),
            })
    return jsonify(positions)


@app.route("/api/equity")
def get_equity():
    """Return equity curve data (mock)."""
    import numpy as np
    n = 60
    base = 10000.0
    drift = np.linspace(0, 8, n)
    noise = np.random.uniform(-3, 3, n)
    values = [round(base + v, 2) for v in (drift + noise)]
    labels = [f"D-{n-i}" for i in range(n)]
    return jsonify({"labels": labels, "values": values})


if __name__ == "__main__":
    app.run(port=5000, debug=False)
