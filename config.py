"""Конфигурация приложения Crypto AI Trader."""

class Config:
    # Путь к базе данных
    DB_PATH = "trades.db"
    
    # ИИ бот
    AI_MODEL_NAME = "gradient_boosting"  # gradient_boosting, random_forest, logistic
    AI_MIN_SAMPLES = 20  # мин. количество сделок для обучения
    
    # Сессии
    SESSIONS = ["Asia", "London", "New York"]
    
    # Таймфреймы
    TIMEFRAMES = ["4H", "1H", "15m", "5m"]
    
    # Типы сетапов
    SETUP_TYPES = [
        "BOS + FVG",
        "CHoCH + OB",
        "MSS + OTE",
        "SMT Divergence",
        "Silver Bullet",
        "Power of 3",
        "Liquidity Sweep",
        "Other"
    ]
    
    # Ключевые уровни/зоны
    ZONE_TYPES = ["Order Block", "FVG", "OTE", "Breaker", "HVN/LVZ"]
    
    # Bias
    BIAS_OPTIONS = ["Long", "Short", "Neutral"]
    
    # Активы
    ASSETS = ["BTC", "ETH", "SOL", "AVAX", "LINK", "ARB", "OP", "DOGE"]
    
    # Риск-параметры
    RISK_PER_TRADE_PCT = 1.0  # % от депозита на сделку
    MAX_DAILY_LOSS_PCT = 3.0   # Макс. дневной убыток %
    POSITION_SIZE_USD = 10000  # Базовый размер позиции

