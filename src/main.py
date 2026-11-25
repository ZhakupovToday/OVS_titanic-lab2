# -*- coding: utf-8 -*-
"""Сервис синхронного инференса: выживет ли пассажир «Титаника»."""
import os
from functools import lru_cache
from typing import Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

import keycloak_utils
import model_utils
from fastapi_utils import Oauth2ClientCredentials


class Passenger(BaseModel):
    """Сырые данные пассажира. age, fare и embarked могут отсутствовать."""
    pclass: Literal[1, 2, 3] = Field(description="Класс билета")
    sex: Literal["male", "female"]
    age: Optional[float] = Field(None, ge=0, le=120, description="Возраст")
    sibsp: int = Field(ge=0, description="Братья/сёстры и супруги на борту")
    parch: int = Field(ge=0, description="Родители и дети на борту")
    fare: Optional[float] = Field(None, ge=0, description="Стоимость билета")
    embarked: Optional[Literal["S", "C", "Q"]] = Field(
        None, description="Порт посадки: Southampton, Cherbourg, Queenstown")

    model_config = {"json_schema_extra": {"examples": [{
        "pclass": 1, "sex": "female", "age": 29, "sibsp": 0, "parch": 0,
        "fare": 211.34, "embarked": "S"}]}}


class Prediction(BaseModel):
    survived: bool
    probability: float = Field(description="Вероятность выживания")


model_path: Optional[str] = os.getenv("MODEL_PATH")
if model_path is None:
    raise ValueError("The environment variable $MODEL_PATH is empty!")

# Ресурс и scope, настроенные в Keycloak для клиента inference-client
PERMISSION = "infer_endpoint#doInfer"

app = FastAPI(title="Titanic survival inference",
              description="Синхронный инференс модели выживания пассажиров. "
                          "Доступ: OAuth 2.0 (client credentials) + Keycloak")
keycloak = keycloak_utils.from_env()
oauth2_scheme = Oauth2ClientCredentials(tokenUrl=keycloak.token_endpoint)


def get_keycloak() -> keycloak_utils.KeycloakClient:
    return keycloak


@lru_cache(maxsize=1)
def get_model() -> Pipeline:
    """Модель загружается один раз, при первом запросе."""
    return model_utils.load_model(model_path)


def check_token(token: str = Depends(oauth2_scheme),
                kc: keycloak_utils.KeycloakClient = Depends(get_keycloak)
                ) -> None:
    """Аутентификация и авторизация клиента через Keycloak.

    Обычная (не async) функция: FastAPI выполнит блокирующий HTTP-запрос
    к Keycloak в пуле потоков и не остановит цикл событий.
    """
    try:
        auth_status = kc.check_access(token, PERMISSION)
    except keycloak_utils.KeycloakError as error:
        print(f"Authorization error: {error}", flush=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authorization server is unavailable",
        )

    if not auth_status.is_logged_in:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not auth_status.is_authorized:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )


@app.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predictions", response_model=Prediction,
          dependencies=[Depends(check_token)])
async def predictions(passenger: Passenger,
                      model: Pipeline = Depends(get_model)) -> Prediction:
    return Prediction(**model_utils.make_inference(model,
                                                   passenger.model_dump()))
