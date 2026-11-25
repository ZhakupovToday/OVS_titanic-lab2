# -*- coding: utf-8 -*-
import os

import joblib
import numpy as np
import pandas as pd
import pytest

from model_utils import FEATURES, load_model, make_inference, to_frame

PASSENGER = {"pclass": 3, "sex": "male", "age": 22.0, "sibsp": 1,
             "parch": 0, "fare": 7.25, "embarked": "S"}


class RecordingModel:
    """Запоминает, какие данные пришли в пайплайн."""

    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.received = None

    def predict_proba(self, frame):
        self.received = frame
        return np.array([[1 - self.probability, self.probability]])


def test_make_inference_passes_raw_data() -> None:
    model = RecordingModel(0.1234)
    result = make_inference(model, PASSENGER)
    assert result == {"survived": False, "probability": 0.123}
    assert list(model.received.columns) == FEATURES
    assert model.received.iloc[0].to_dict() == PASSENGER


@pytest.mark.parametrize("probability, survived",
                         [(0.5, True), (0.49, False), (0.99, True)])
def test_threshold(probability, survived) -> None:
    result = make_inference(RecordingModel(probability), PASSENGER)
    assert result["survived"] is survived


def test_missing_values_become_nan() -> None:
    frame = to_frame({"pclass": 2, "sex": "female", "sibsp": 0, "parch": 0,
                      "age": None, "fare": None, "embarked": None})
    assert np.isnan(frame.loc[0, "age"])
    assert np.isnan(frame.loc[0, "fare"])
    assert pd.isna(frame.loc[0, "embarked"])


def test_load_model(tmp_path) -> None:
    path = os.path.join(tmp_path, "model.pkl")
    joblib.dump({"message": "Test"}, path)
    assert load_model(path) == {"message": "Test"}


def test_trained_pipeline_accepts_raw_data(tmp_path) -> None:
    """Обучаем настоящий пайплайн и проверяем его на сырых данных."""
    import train
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    x, y = train.load_data(os.path.join(root, "data", "titanic.csv"))
    pipeline = train.build_pipeline().fit(x, y)
    path = os.path.join(tmp_path, "pipeline.pkl")
    joblib.dump(pipeline, path)

    model = load_model(path)
    rich_woman = {"pclass": 1, "sex": "female", "age": 30, "sibsp": 0,
                  "parch": 0, "fare": 100.0, "embarked": "C"}
    poor_man = {"pclass": 3, "sex": "male", "age": 30, "sibsp": 0,
                "parch": 0, "fare": 7.0, "embarked": "S"}
    unknown = {"pclass": 2, "sex": "male", "sibsp": 0, "parch": 0}
    assert make_inference(model, rich_woman)["survived"] is True
    assert make_inference(model, poor_man)["survived"] is False
    assert 0 <= make_inference(model, unknown)["probability"] <= 1
