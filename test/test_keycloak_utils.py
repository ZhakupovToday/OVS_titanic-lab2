# -*- coding: utf-8 -*-
import pytest
import requests

import keycloak_utils
from keycloak_utils import KeycloakClient, KeycloakError


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self.payload = payload or {}
        self.text = str(self.payload)

    def json(self) -> dict:
        return self.payload


@pytest.fixture
def kc() -> KeycloakClient:
    return KeycloakClient("https://keycloak:8443/", "inference",
                          "inference-client", "secret")


def mock_post(monkeypatch, response=None, error=None) -> dict:
    calls = {}

    def fake_post(url, **kwargs):
        calls.update(url=url, **kwargs)
        if error:
            raise error
        return response

    monkeypatch.setattr(keycloak_utils.requests, "post", fake_post)
    return calls


def test_token_endpoint(kc) -> None:
    assert kc.token_endpoint == (
        "https://keycloak:8443/realms/inference/protocol/openid-connect/token")


def test_uma_request(monkeypatch, kc) -> None:
    calls = mock_post(monkeypatch, FakeResponse(200, {"result": True}))
    kc.check_access("abc", "infer_endpoint#doInfer")
    assert calls["url"] == kc.token_endpoint
    assert calls["headers"] == {"Authorization": "Bearer abc"}
    assert calls["data"] == {
        "grant_type": "urn:ietf:params:oauth:grant-type:uma-ticket",
        "audience": "inference-client",
        "permission": "infer_endpoint#doInfer",
        "response_mode": "decision"}


@pytest.mark.parametrize("code, payload, logged, authorized", [
    (200, {"result": True}, True, True),
    (200, {"result": False}, True, False),
    (403, {"error": "access_denied"}, True, False),
    (401, {"error": "invalid_grant"}, False, False),
])
def test_access_status(monkeypatch, kc, code, payload, logged,
                       authorized) -> None:
    mock_post(monkeypatch, FakeResponse(code, payload))
    result = kc.check_access("abc", "infer_endpoint#doInfer")
    assert result.is_logged_in is logged
    assert result.is_authorized is authorized


def test_unexpected_response(monkeypatch, kc) -> None:
    mock_post(monkeypatch, FakeResponse(400, {"error": "invalid_resource"}))
    with pytest.raises(KeycloakError):
        kc.check_access("abc", "infer_endpoint#doInfer")


def test_keycloak_unreachable(monkeypatch, kc) -> None:
    mock_post(monkeypatch, error=requests.ConnectionError("refused"))
    with pytest.raises(KeycloakError):
        kc.check_access("abc", "infer_endpoint#doInfer")


def test_from_env_requires_settings(monkeypatch) -> None:
    for name in ("KEYCLOAK_URL", "CLIENT_ID", "CLIENT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError):
        keycloak_utils.from_env()
    monkeypatch.setenv("KEYCLOAK_URL", "https://keycloak:8443")
    with pytest.raises(ValueError):
        keycloak_utils.from_env()


def test_from_env(monkeypatch) -> None:
    monkeypatch.setenv("KEYCLOAK_URL", "https://keycloak:8443")
    monkeypatch.setenv("CLIENT_ID", "inference-client")
    monkeypatch.setenv("CLIENT_SECRET", "secret")
    monkeypatch.setenv("KEYCLOAK_REALM", "titanic")
    kc = keycloak_utils.from_env()
    assert kc.client_id == "inference-client"
    assert "/realms/titanic/" in kc.token_endpoint
    assert kc.verify_ssl is False
