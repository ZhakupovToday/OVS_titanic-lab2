# -*- coding: utf-8 -*-
"""Загрузка сохранённого пайплайна и инференс."""
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

NUMERIC_FEATURES = ["age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["pclass", "sex", "embarked"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def to_frame(in_data: dict) -> pd.DataFrame:
    """Превращает сырые данные одного пассажира в DataFrame для пайплайна.

    Пропущенные значения (None) заменяются на NaN, чтобы их обработали
    импьютеры внутри пайплайна.
    """
    frame = pd.DataFrame([{name: in_data.get(name) for name in FEATURES}])
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype(float)
    frame[CATEGORICAL_FEATURES] = frame[CATEGORICAL_FEATURES].astype(object)
    return frame.replace({None: np.nan})


def make_inference(in_model: Pipeline, in_data: dict) -> dict:
    """Возвращает предсказание: выжил ли пассажир и вероятность выживания."""
    probability = float(in_model.predict_proba(to_frame(in_data))[0][1])
    return {"survived": probability >= 0.5,
            "probability": round(probability, 3)}


def load_model(path: str) -> Pipeline:
    """Читает сохранённый пайплайн (предобработка + модель) из файла."""
    return joblib.load(path)
