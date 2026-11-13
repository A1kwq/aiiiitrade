"""Слой работы с базой данных (SQLite)."""

import sqlite3
from datetime import datetime
from config import Config


def get_connection():
    conn = sqlite3.connect(Config.DB_PATH)
    return conn


def init_db():
    """Инициализация таблиц."""
    conn = get_connection()
    cur = conn.cursor()
    
    # Таблица сделок
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            asset TEXT NOT NULL,
            direction TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            session TEXT,
            setup_type TEXT,
            bias TEXT,
            entry_price REAL,
            exit_price REAL,
            stop_loss REAL,
            position_size REAL,
            r_multiple REAL,
            pnl_usd REAL,
            pnl_pct REAL,
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    
    # Таблица метрик для ИИ
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ai_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER,
            feature_name TEXT,
            feature_value REAL,
            is_success INTEGER
        )
    """)
    
    conn.commit()
    conn.close()


def add_trade(data: dict) -> int:
    """Добавить сделку."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        f"INSERT INTO trades ({', '.join(data.keys())}) VALUES ({''.join(['?,' for _ in data])})",
        tuple(data.values())
    )
    trade_id = cur.lastrowid
    conn.commit()
    conn.close()
    return trade_id


def get_trades(start_date: str = None, end_date: str = None) -> list[dict]:
    """Получить сделки за период."""
    conn = get_connection()
    cur = conn.cursor()
    query = "SELECT * FROM trades WHERE 1=1"
    if start_date:
        query += f" AND trade_date >= '{start_date}'"
    if end_date:
        query += f" AND trade_date <= '{end_date}'"
    query += " ORDER BY trade_date DESC"
    cur.execute(query)
    rows = [dict(zip([d[0] for d in cur.description], row)) for row in cur.fetchall()]
    conn.close()
    return rows


def delete_trade(trade_id: int):
    """Удалить сделку."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
    conn.commit()
    conn.close()


def get_stats() -> dict:
    """Базовая статистика."""
    trades = get_trades()
    if not trades:
        return {"total": 0}
    
    total = len(trades)
    wins = sum(1 for t in trades if t["r_multiple"] > 0)
    losses = total - wins
    win_rate = (wins / total * 100) if total else 0
    avg_r = sum(t["r_multiple"] for t in trades) / total if total else 0
    total_pnl = sum(t["pnl_usd"] for t in trades)
    
    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "avg_r": round(avg_r, 2),
        "total_pnl": round(total_pnl, 2),
    }


class Database:
    """Класс-обёртка для работы с базой."""

    def __init__(self):
        init_db()

    def add_trade(self, data: dict) -> int:
        return add_trade(data)

    def get_all_trades(self) -> list[dict]:
        return get_trades()

    def delete_trade(self, trade_id: int):
        delete_trade(trade_id)

    def get_stats(self) -> dict:
        return get_stats()
