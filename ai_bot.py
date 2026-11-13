"""Обучаемый ИИ-бот для оценки сделок."""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from config import Config


class AIBot:
    """Обучаемый бот для оценки вероятности успеха сетапа."""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.feature_names = []
        
    def _get_model(self, name: str):
        if name == "gradient_boosting":
            return GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif name == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            return LogisticRegression(max_iter=1000)
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Подготовить фичи для обучения."""
        # Фичи: направление, таймфрейм, сессия, тип сетапа, bias
        feature_cols = ["direction", "timeframe", "session", "setup_type", "bias"]
        
        # One-hot encode
        df_feat = pd.get_dummies(df[feature_cols].astype(str), columns=feature_cols, drop_first=True)
        
        # Целевая переменная: успех (r_multiple > 0)
        y = (df["r_multiple"] > 0).astype(int)
        
        return df_feat, y
    
    def train(self, df: pd.DataFrame) -> dict:
        """Обучить модель на исторических данных."""
        if len(df) < Config.AI_MIN_SAMPLES:
            return {"status": "not_enough_data", "need": Config.AI_MIN_SAMPLES, "have": len(df)}
        
        X, y = self.prepare_features(df)
        
        # Убрать строки с NaN
        mask = ~X.isna().any(axis=1) & ~y.isna()
        X, y = X[mask], y[mask]
        
        if len(X) < Config.AI_MIN_SAMPLES:
            return {"status": "not_enough_data", "need": Config.AI_MIN_SAMPLES, "have": len(X)}
        
        self.model = self._get_model(Config.AI_MODEL_NAME)
        self.feature_names = list(X.columns)
        
        # Кросс-валидация для оценки качества
        scores = cross_val_score(self.model, X, y, cv=min(5, len(X)), scoring="accuracy")
        accuracy = scores.mean()
        
        self.model.fit(X, y)
        self.is_trained = True
        
        return {
            "status": "trained",
            "accuracy": round(accuracy * 100, 1),
            "samples": len(X),
            "model": Config.AI_MODEL_NAME
        }
    
    def predict(self, direction: str, timeframe: str, session: str, 
               setup_type: str, bias: str) -> float:
        """Оценить вероятность успеха сетапа."""
        if not self.is_trained or self.model is None:
            return 0.5  # нейтральная оценка
        
        input_df = pd.DataFrame([{
            "direction": direction,
            "timeframe": timeframe,
            "session": session,
            "setup_type": setup_type,
            "bias": bias
        }])
        
        X_input = pd.get_dummies(input_df.astype(str), columns=list(input_df.columns), drop_first=True)
        
        # Привести к тем же колонкам что при обучении
        for col in self.feature_names:
            if col not in X_input.columns:
                X_input[col] = 0
        
        X_input = X_input[self.feature_names]
        
        prob = self.model.predict_proba(X_input)[0][1]
        return round(prob, 3)
    
    def get_feature_importance(self) -> dict:
        """Важность фичей (для объяснимости)."""
        if not self.is_trained or self.model is None:
            return {}
        
        importances = {}
        if hasattr(self.model, "feature_importances_"):
            for name, imp in zip(self.feature_names, self.model.feature_importances_):
                importances[name] = round(imp, 4)
        elif hasattr(self.model, "coef_"):
            for i, name in enumerate(self.feature_names):
                importances[name] = round(abs(self.model.coef_[0][i]), 4)
        
        return dict(sorted(importances.items(), key=lambda x: -x[1]))


# Глобальный инстанс
bot = AIBot()
