# -*- coding: utf-8 -*-
"""Обучение модели выживания пассажиров «Титаника».

Сохраняет ВЕСЬ пайплайн (импьютеры, масштабирование, one-hot кодирование
и классификатор) в один файл, поэтому сервис принимает сырые данные.

Запуск:  python src/train.py --data data/titanic.csv --out models/pipeline.pkl
"""
import argparse
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_utils import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

TARGET = "survived"
RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    classifier = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        random_state=RANDOM_STATE)
    return Pipeline([("preprocessing", preprocessing),
                     ("classifier", classifier)])


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    features = data[FEATURES].copy()
    features[CATEGORICAL_FEATURES] = features[CATEGORICAL_FEATURES].astype(
        object)
    return features, data[TARGET]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/titanic.csv")
    parser.add_argument("--out", default="models/pipeline.pkl")
    args = parser.parse_args()

    x, y = load_data(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    pipeline = build_pipeline()
    cv = cross_val_score(pipeline, x_train, y_train, cv=5, scoring="accuracy")
    print(f"CV accuracy: {cv.mean():.3f} ± {cv.std():.3f}")

    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    probability = pipeline.predict_proba(x_test)[:, 1]
    print(f"Test accuracy: {accuracy_score(y_test, predicted):.3f}")
    print(f"Test ROC AUC:  {roc_auc_score(y_test, probability):.3f}")
    print(classification_report(y_test, predicted, digits=3))

    # Финальная модель обучается на всех данных
    pipeline.fit(x, y)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(pipeline, args.out)
    print(f"Pipeline saved to {args.out}")


if __name__ == "__main__":
    main()
