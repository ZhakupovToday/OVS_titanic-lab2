# Лабораторная работа 2. Контейнеризация, аутентификация и авторизация сервиса

Сервис инференса из лабораторной 1 (выживание пассажиров «Титаника») доработан:

- доступ к `/predictions` защищён по OAuth 2.0 (client credentials), решения о доступе принимает **Keycloak** (Authorization Services, UMA 2.0);
- сервис упакован в Docker-образ, модель обучается прямо при сборке образа;
- запуск всего стенда (PostgreSQL → Keycloak → inference) идёт через `docker-compose`;
- CI/CD в GitHub Actions: pycodestyle + pytest → сборка образов → публикация в DockerHub.

## Структура

```
src/main.py            FastAPI-сервис
src/keycloak_utils.py  запрос решения у Keycloak (UMA, response_mode=decision)
src/fastapi_utils.py   схема безопасности OAuth2 Client Credentials для OpenAPI
src/model_utils.py     загрузка пайплайна и инференс
src/train.py           обучение пайплайна
test/                  юнит-тесты (Keycloak и модель заменены заглушками)
docker/Dockerfile      многоэтапная сборка: зависимости → обучение → рабочий образ
keycloak/docker/       образ Keycloak 21 c TLS-сертификатом и PostgreSQL
docker-compose.yml     оркестрация postgres, keycloak, inference
.github/workflows/ci-cd.yml
```

## Как проверяется доступ

1. Клиент (`privileged-client` или `unprivileged-client`) получает в Keycloak access token по grant type `client_credentials`.
2. Клиент вызывает `POST /predictions` с заголовком `Authorization: Bearer <token>`.
3. Сервис пересылает токен в Keycloak (`grant_type=urn:ietf:params:oauth:grant-type:uma-ticket`, `permission=infer_endpoint#doInfer`, `response_mode=decision`) и по ответу решает:

| Ответ Keycloak | Ответ сервиса |
|---|---|
| `200 {"result": true}` | `200` + предсказание |
| `403` (политика запрещает доступ) | `403 Access denied` |
| `401` (токен недействителен) | `401 Invalid authentication credentials` |
| Keycloak недоступен или ответил иначе | `503 Authorization server is unavailable` |
| Нет заголовка Authorization | `401 Not authenticated` |
| Неверные данные пассажира | `422` |

## Запуск

```bash
cp .env.example .env              # затем задать пароли
docker compose up -d --build postgres keycloak
docker compose logs -f keycloak   # дождаться "Keycloak ... started"
```

### Настройка Keycloak

Консоль администратора: https://localhost:8443. Сертификат самоподписанный, браузер нужно попросить продолжить. Вход: `admin` / `KEYCLOAK_ADMIN_PASSWORD` из `.env`.

1. **Создать realm** `inference`, настройки оставить по умолчанию.
2. **Clients → Create client** → `inference-client`. На шаге Capability config включить *Client authentication*, *Authorization* и *Service accounts roles*. В Login settings указать *Web origins* = `*`.
   Затем **Credentials** → скопировать *Client secret* в `.env` (`CLIENT_SECRET`).
3. Так же создать `privileged-client` и `unprivileged-client` и сохранить их секреты.
4. В `inference-client` → **Authorization**:
   1. **Scopes → Create authorization scope**: `doInfer`.
   2. **Resources → Create resource**: имя `infer_endpoint`, URI `/predictions`, scope `doInfer`.
   3. **Policies → Create client policy** (тип *Client*): имя `inference-policy`, клиент `privileged-client`, логика *Positive*.
   4. **Permissions → Create scope-based permission**: имя `inference-permission`, ресурс `infer_endpoint`, scope `doInfer`, политика `inference-policy`, стратегия *Unanimous*.
5. Проверить: **Authorization → Evaluate**. Для `privileged-client` результат должен быть PERMIT, для `unprivileged-client` — DENY.

Затем запустить сервис:

```bash
docker compose up -d --build inference
curl http://localhost:8000/healthcheck          # {"status":"ok"}
```

Если позже изменился `CLIENT_SECRET`, выполните `docker compose up -d inference`, и контейнер пересоздастся с новыми переменными.

### Проверка запросами

```bash
# Токен привилегированного клиента
TOKEN=$(curl -sk -X POST https://localhost:8443/realms/inference/protocol/openid-connect/token \
  -d grant_type=client_credentials -d client_id=privileged-client \
  -d client_secret=<секрет privileged-client> | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

curl -X POST http://localhost:8000/predictions \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"pclass": 1, "sex": "female", "age": 29, "sibsp": 0, "parch": 0, "fare": 211.34, "embarked": "S"}'
# → 200 {"survived":true,"probability":0.95...}

# То же для unprivileged-client → 403 {"detail":"Access denied"}
# Испорченный токен (например, "Bearer abc") → 401 {"detail":"Invalid authentication credentials"}
# Без заголовка Authorization → 401 {"detail":"Not authenticated"}
```

Токен живёт 5 минут (настройка realm по умолчанию). Если пришёл 401, получите новый токен.

**Если что-то не так:**
- `503`: смотрите `docker compose logs inference`, там будет ответ Keycloak. Чаще всего причина в неверном `CLIENT_SECRET`, в другом имени realm, ресурса или scope, или в том, что у `inference-client` не включён *Authorization*.
- `unprivileged-client` получает `200`: проверьте политику в `inference-policy` и стратегию *Unanimous* в разрешении. Если в клиенте остались автоматически созданные *Default Policy* и *Default Permission*, их можно удалить.

Логи и статус: `docker compose ps`, `docker compose logs inference`. Остановить стенд: `docker compose down`; добавьте `-v`, чтобы удалить и базу Keycloak.

## Тесты локально

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pycodestyle src test
PYTHONPATH=./:./src pytest -v test
```

## CI/CD

Workflow `.github/workflows/ci-cd.yml`:

- `tests` запускается при каждом push и pull request в `main`: линтер и юнит-тесты;
- `build-and-push` запускается только после успешных тестов и только при push в `main`. Он собирает и публикует образы `<user>/titanic-inference` и `<user>/titanic-keycloak` с тегами `latest` и SHA коммита.

Нужные секреты: **Settings → Secrets and variables → Actions**:

| Секрет | Значение |
|---|---|
| `DOCKERHUB_USERNAME` | логин DockerHub |
| `DOCKERHUB_TOKEN` | Access Token: DockerHub → Account settings → Personal access tokens, права Read & Write |

Чтобы запустить стенд из опубликованных образов без сборки, задайте `DOCKERHUB_USERNAME` в `.env`, затем выполните `docker compose pull inference keycloak` и `docker compose up -d`.
