# -*- coding: utf-8 -*-
"""Проверка прав клиента через Keycloak Authorization Services (UMA 2.0).

Сервис отправляет в Keycloak токен клиента и запрашивает решение
(response_mode=decision) по разрешению ``infer_endpoint#doInfer``:
  200 {"result": true} -> доступ разрешён;
  403                  -> токен валиден, но политика запрещает доступ;
  401                  -> токен недействителен (истёк, подделан и т. п.).
"""
import os
from dataclasses import dataclass

import requests
import urllib3

UMA_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:uma-ticket"


class KeycloakError(Exception):
    """Keycloak недоступен или настроен неверно."""


@dataclass(frozen=True)
class AccessStatus:
    is_logged_in: bool
    is_authorized: bool


class KeycloakClient:
    def __init__(self, server_url: str, realm: str, client_id: str,
                 client_secret: str, verify_ssl: bool = False,
                 timeout: float = 5.0) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.token_endpoint = (f"{server_url.rstrip('/')}/realms/{realm}"
                               "/protocol/openid-connect/token")
        if not verify_ssl:
            # В учебном стенде у Keycloak самоподписанный сертификат
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def check_access(self, token: str, permission: str) -> AccessStatus:
        try:
            response = requests.post(
                self.token_endpoint,
                headers={"Authorization": f"Bearer {token}"},
                data={"grant_type": UMA_GRANT_TYPE,
                      "audience": self.client_id,
                      "permission": permission,
                      "response_mode": "decision"},
                verify=self.verify_ssl,
                timeout=self.timeout,
            )
        except requests.RequestException as error:
            raise KeycloakError(f"Keycloak is unreachable: {error}")

        if response.status_code == 200:
            granted = bool(response.json().get("result", False))
            return AccessStatus(is_logged_in=True, is_authorized=granted)
        if response.status_code == 403:
            return AccessStatus(is_logged_in=True, is_authorized=False)
        if response.status_code == 401:
            return AccessStatus(is_logged_in=False, is_authorized=False)
        raise KeycloakError(
            f"Unexpected Keycloak response {response.status_code}: "
            f"{response.text}")


def from_env() -> KeycloakClient:
    keycloak_url = os.getenv("KEYCLOAK_URL")
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")

    if keycloak_url is None:
        raise ValueError("The Keycloak URL isn't defined!")
    if client_id is None or client_secret is None:
        raise ValueError("The client's credentials aren't defined!")

    return KeycloakClient(
        server_url=keycloak_url,
        realm=os.getenv("KEYCLOAK_REALM", "inference"),
        client_id=client_id,
        client_secret=client_secret,
        verify_ssl=os.getenv("KEYCLOAK_VERIFY_SSL", "false").lower()
        == "true",
    )
