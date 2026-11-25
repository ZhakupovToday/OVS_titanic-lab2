# -*- coding: utf-8 -*-
import numpy as np
import pytest
from fastapi.testclient import TestClient

from keycloak_utils import AccessStatus, KeycloakError

PASSENGER = {"pclass": 1, "sex": "female", "age": 29, "sibsp": 0,
             "parch": 0, "fare": 211.34, "embarked": "S"}


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class FakeModel:
    def predict_proba(self, frame):
        return np.array([[0.07, 0.93]])


class FakeKeycloak:
    """Имитирует ответы Keycloak для разных токенов."""
    permissions: list[str] = []

    def check_access(self, token: str, permission: str) -> AccessStatus:
        self.permissions.append(permission)
        if token == "privileged":
            return AccessStatus(is_logged_in=True, is_authorized=True)
        if token == "unprivileged":
            return AccessStatus(is_logged_in=True, is_authorized=False)
        if token == "keycloak_down":
            raise KeycloakError("connection refused")
        return AccessStatus(is_logged_in=False, is_authorized=False)


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("MODEL_PATH", "faked/model.pkl")
    monkeypatch.setenv("KEYCLOAK_URL", "https://keycloak:8443")
    monkeypatch.setenv("CLIENT_ID", "inference-client")
    monkeypatch.setenv("CLIENT_SECRET", "secret")
    import main
    main.app.dependency_overrides[main.get_model] = FakeModel
    main.app.dependency_overrides[main.get_keycloak] = FakeKeycloak
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_healthcheck(client) -> None:
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_privileged_client(client) -> None:
    response = client.post("/predictions", headers=bearer("privileged"),
                           json=PASSENGER)
    assert response.status_code == 200
    assert response.json() == {"survived": True, "probability": 0.93}
    assert FakeKeycloak.permissions[-1] == "infer_endpoint#doInfer"


def test_unprivileged_client(client) -> None:
    response = client.post("/predictions", headers=bearer("unprivileged"),
                           json=PASSENGER)
    assert response.status_code == 403
    assert response.json() == {"detail": "Access denied"}


def test_invalid_token(client) -> None:
    response = client.post("/predictions", headers=bearer("forged"),
                           json=PASSENGER)
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid authentication credentials"}


def test_token_absent(client) -> None:
    response = client.post("/predictions", json=PASSENGER)
    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}


def test_keycloak_unavailable(client) -> None:
    response = client.post("/predictions", headers=bearer("keycloak_down"),
                           json=PASSENGER)
    assert response.status_code == 503


@pytest.mark.parametrize("field, value", [
    ("pclass", 4), ("sex", "unknown"), ("age", -5), ("embarked", "X"),
])
def test_invalid_input(client, field, value) -> None:
    response = client.post("/predictions", headers=bearer("privileged"),
                           json={**PASSENGER, field: value})
    assert response.status_code == 422


def test_openapi_has_client_credentials_flow(client) -> None:
    schema = client.get("/openapi.json").json()
    flows = schema["components"]["securitySchemes"][
        "Oauth2ClientCredentials"]["flows"]
    assert flows["clientCredentials"]["tokenUrl"] == (
        "https://keycloak:8443/realms/inference/protocol/openid-connect/token")
