# -*- coding: utf-8 -*-
"""Схема безопасности OAuth 2.0 Client Credentials для FastAPI.

В FastAPI нет готового класса для grant type client_credentials, поэтому
он описан здесь: извлекает Bearer-токен из заголовка Authorization и
публикует tokenUrl в OpenAPI (кнопка Authorize в /docs).
"""
from typing import Optional

from fastapi import HTTPException, status
from fastapi.openapi.models import OAuthFlows as OAuthFlowsModel
from fastapi.security import OAuth2
from fastapi.security.utils import get_authorization_scheme_param
from starlette.requests import Request


class Oauth2ClientCredentials(OAuth2):
    def __init__(self, tokenUrl: str, scheme_name: Optional[str] = None,
                 scopes: Optional[dict] = None,
                 auto_error: bool = True) -> None:
        flows = OAuthFlowsModel(clientCredentials={"tokenUrl": tokenUrl,
                                                   "scopes": scopes or {}})
        super().__init__(flows=flows, scheme_name=scheme_name,
                         auto_error=auto_error)

    async def __call__(self, request: Request) -> Optional[str]:
        authorization = request.headers.get("Authorization")
        scheme, param = get_authorization_scheme_param(authorization)
        if not authorization or scheme.lower() != "bearer":
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return None
        return param
