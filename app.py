"""Crypto AI Trader — торговый журнал с обучаемым ИИ-ботом."""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from config import Config
from database import Database
from ai_bot import bot


# ─── Инициализация ──────────────────────────────────────────────
db = Database()

# ─── Стили (StatsAI-подобный дизайн) ───────────────────────────
st.markdown("""
<style>
/* StatsAI-inspired dark theme */
.stApp { background-color: #0d1117; }
h1, h2, h3 { color: #ffffff !important; }
p, li { color: #b0b7c3; }
.stButton > button {
    background-color: #1a9144 !important;
    color: white !important;
    border-radius: 6px;
    padding: 8px 20px;
}
.stButton > button:hover { background-color: #21a557 !important; }
div[data-testid="stMetric"] {
    background-color: #1e2534;
    border-radius: 8px;
    padding: 12px;
    border: 1px solid #2d3746;
}
div[data-testid="stMetric"] label { color: #6b7a8f !important; }
div[data-testid="stMetric"] p { color: #ffffff !important; }
.stTabs [role="tab"] { color: #6b7a8f !important; }
.stTabs [role="tab"][aria-selected="true"] { color: #1a9144 !important; }
</style>
""")

st.set_page_config(page_title="Crypto AI Trader", layout="wide")


# ─── Хелперы ─────────────────────────────────────────────────────
def get_trades_df() -> pd.DataFrame:
    trades = db.get_all_trades()
    if not trades:
        return pd.DataFrame(columns=["id", "entry_date", "symbol", "direction", "timeframe",
                                     "session", "setup_type", "bias", "entry_price",
                                     "exit_price", "stop_loss", "position_size",
                                     "r_multiple", "notes"])
    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    return df


def calc_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    
    total = len(df)
    wins = (df["r_multiple"] > 0).sum()
    losses = (df["r_multiple"] < 0).sum()
    win_rate = (wins / total * 100) if total else 0
    
    total_pnl = df["r_multiple"].sum()
    avg_r = df["r_multiple"].mean()
    
    # Drawdown
    cum = df["r_multiple"].cumsum()
    max_dd = (cum - cum.cummax()).min()
    
    return {
        "total_trades": total,
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": round(win_rate, 1),
        "total_pnl_r": round(total_pnl, 2),
        "avg_r": round(avg_r, 3),
        "max_drawdown": round(max_dd, 2)
    }


def calc_session_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    stats = {}
    for sess in ["asia", "europe", "us"]:
        sub = df[df["session"] == sess]
        if not sub.empty:
            stats[sess] = {
                "trades": len(sub),
                "win_rate": round((sub["r_multiple"] > 0).sum() / len(sub) * 100, 1),
                "pnl_r": round(sub["r_multiple"].sum(), 2)
            }
    return stats


def calc_setup_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    stats = {}
    for setup in Config.SETUP_TYPES:
        sub = df[df["setup_type"] == setup]
        if not sub.empty:
            stats[setup] = {
                "trades": len(sub),
                "win_rate": round((sub["r_multiple"] > 0).sum() / len(sub) * 100, 1),
                "pnl_r": round(sub["r_multiple"].sum(), 2)
            }
    return stats


# ─── Сайдбар ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Crypto AI Trader")
    st.markdown("---")
    
    tab = st.radio("Навигация", ["Дашборд", "Новая сделка", "ИИ-бот", "Статистика"], 
                  format_func=str)
    
    st.markdown("---")
    st.markdown(f"**Сделок в журнале:** {len(get_trades_df())}")
    st.markdown(f"**Модель ИИ:** {'Обучена' if bot.is_trained else 'Не обучена'}")


# ─── ДАШБОРД ─────────────────────────────────────────────────────
if tab == "Дашборд":
    df = get_trades_df()
    
    st.markdown("## 📈 Обзор")
    
    if df.empty:
        st.info("Добавь первую сделку во вкладке **Новая сделка**.")
    else:
        stats = calc_stats(df)
        
        # KPI карточки
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Сделок", stats["total_trades"])
        with col2:
            st.metric("Win Rate", f"{stats['win_rate']}%")
        with col3:
            st.metric("P&L (R)", f"{stats['total_pnl_r']:+.1f}R")
        with col4:
            st.metric("Avg R", f"{stats['avg_r']:.2f}")
        with col5:
            st.metric("Max DD (R)", f"{stats['max_drawdown']:.1f}")
        
        # График P&L
        if len(df) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["entry_date"], 
                y=df["r_multiple"].cumsum(),
                mode="lines",
                line=dict(color="#1a9144", width=2),
                name="Cumulative P&L (R)"
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="#374151")
            fig.update_layout(
                title="Накопительный P&L (R)",
                xaxis_title="", yaxis_title="R",
                template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Сессии
        session_stats = calc_session_stats(df)
        if session_stats:
            st.markdown("### По сессиям")
            cols = st.columns(len(session_stats))
            for i, (sess, s) in enumerate(session_stats.items()):
                with cols[i]:
                    st.metric(f"📡 {sess.capitalize()}", 
                              f"{s['win_rate']}% | {s['pnl_r']:+.1f}R",
                              delta=f"{s['trades']} сделок")
        
        # Сетапы
        setup_stats = calc_setup_stats(df)
        if setup_stats:
            st.markdown("### По сетапам")
            for setup, s in setup_stats.items():
                st.markdown(f"- **{setup}**: {s['trades']} сделок, WR {s['win_rate']}%, P&L {s['pnl_r']:+.1f}R")


# ─── НОВАЯ СДЕЛКА ────────────────────────────────────────────────
elif tab == "Новая сделка":
    st.markdown("## ➕ Запись сделки")
    
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Символ", value="SOL/USDT")
        direction = st.selectbox("Направление", Config.DIRECTIONS)
        timeframe = st.selectbox("Таймфрейм", Config.TIMEFRAMES)
        session = st.selectbox("Сессия", Config.SESSIONS)
    with col2:
        setup_type = st.selectbox("Тип сетапа", Config.SETUP_TYPES)
        bias = st.selectbox("Bias", ["long", "short", "neutral"])
        entry_price = st.number_input("Entry Price", min_value=0.0, format="%.4f")
        stop_loss = st.number_input("Stop Loss", min_value=0.0, format="%.4f")
    
    col3, col4, col5 = st.columns(3)
    with col3:
        position_size = st.number_input("Позиция (R)", min_value=0.1, step=0.1)
    with col4:
        exit_price = st.number_input("Exit Price", min_value=0.0, format="%.4f")
    with col5:
        notes = st.text_area("Заметки / Order Flow / GEX")
    
    if st.button("Сохранить сделку"):
        entry_date = datetime.now().isoformat()
        r_multiple = (exit_price - entry_price) / (entry_price - stop_loss) if direction == "long"                     else (entry_price - exit_price) / (stop_loss - entry_price)
        
        db.add_trade({
            "entry_date": entry_date,
            "symbol": symbol,
            "direction": direction,
            "timeframe": timeframe,
            "session": session,
            "setup_type": setup_type,
            "bias": bias,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stop_loss": stop_loss,
            "position_size": position_size,
            "r_multiple": round(r_multiple, 2),
            "notes": notes
        })
        st.success(f"Сделка сохранена! R: {r_multiple:+.1f}")


# ─── ИИ-БОТ ───────────────────────────────────────────────────────
elif tab == "ИИ-бот":
    st.markdown("## 🤖 Обучаемый ИИ-бот")
    
    df = get_trades_df()
    
    if df.empty:
        st.info("Нужны минимум 10 сделок для обучения бота.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            model_name = st.selectbox("Модель", 
                ["gradient_boosting", "random_forest", "logistic"], 
                index=0)
        with col2:
            train_btn = st.button("🎓 Обучить")
        with col3:
            st.metric("Сделок для обучения", len(df))
        
        if train_btn:
            result = bot.train(df)
            if result["status"] == "trained":
                st.success(f"✅ Модель обучена! Accuracy: {result['accuracy']}%")
            else:
                st.warning(f"Недостаточно данных: {result['have']}/{result['need']}")
        
        # Оценка сетапа
        if bot.is_trained:
            st.markdown("### Оценка нового сетапа")
            eval_col1, eval_col2, eval_col3, eval_col4, eval_col5 = st.columns(5)
            with eval_col1:
                d = st.selectbox("Direction", Config.DIRECTIONS, key="d")
            with eval_col2:
                tf = st.selectbox("Timeframe", Config.TIMEFRAMES, key="tf")
            with eval_col3:
                se = st.selectbox("Session", Config.SESSIONS, key="se")
            with eval_col4:
                stp = st.selectbox("Setup", Config.SETUP_TYPES, key="stp")
            with eval_col5:
                b = st.selectbox("Bias", ["long", "short", "neutral"], key="b")
            
            prob = bot.predict(d, tf, se, stp, b)
            st.markdown(f"### 📊 Вероятность успеха: **{prob*100:.0f}%**")
            
            if prob < 0.4:
                st.warning("⚠️ Низкая вероятность — лучше пропустить.")
            elif prob > 0.65:
                st.success("✅ Высокая вероятность — сетап выглядит сильно.")
            else:
                st.info("📐 Средняя вероятность — ищи подтверждение order flow.")
        
        # Важность фичей
        if bot.is_trained:
            importance = bot.get_feature_importance()
            if importance:
                st.markdown("### Важность фичей (что влияет на результат):")
                for feat, imp in importance.items():
                    st.progress(imp * 100, text=f"{feat}: {imp:.2f}")


# ─── СТАТИСТИКА ───────────────────────────────────────────────────
elif tab == "Статистика":
    st.markdown("## 📊 Детальная статистика")
    
    df = get_trades_df()
    
    if df.empty:
        st.info("Нет данных для анализа.")
    else:
        stats = calc_stats(df)
        
        # Сессии
        session_stats = calc_session_stats(df)
        st.markdown("### Win Rate по сессиям")
        if session_stats:
            fig = go.Figure()
            for sess, s in session_stats.items():
                fig.add_bar(x=[sess], y=[s["win_rate"]], 
                           name=sess, marker_color="#1a9144")
            fig.update_layout(template="plotly_dark", 
                             title="Win Rate % по сессиям",
                             paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
            st.plotly_chart(fig, use_container_width=True)
        
        # Сетапы
        setup_stats = calc_setup_stats(df)
        st.markdown("### Win Rate по сетапам")
        if setup_stats:
            fig2 = go.Figure()
            for setup, s in setup_stats.items():
                fig2.add_bar(x=[setup], y=[s["win_rate"]], 
                            name=setup, marker_color="#3b82f6")
            fig2.update_layout(template="plotly_dark", 
                              title="Win Rate % по сетапам",
                              paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Тейбл сделок
        st.markdown("### Все сделки")
        display_df = df[["entry_date", "symbol", "direction", "setup_type", 
                        "r_multiple", "notes"]].copy()
        display_df.columns = ["Дата", "Символ", "Направление", "Сетап", "R", "Заметки"]
        st.dataframe(display_df, use_container_width=True)
