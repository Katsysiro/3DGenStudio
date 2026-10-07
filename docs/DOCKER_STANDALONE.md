# 3D Gen Studio в Docker вместе с ComfyUI из другого контейнера

## Почему мастер установки «не идёт дальше» с родным docker-compose.yml

Штатные `Dockerfile` + `docker-compose.yml` собирают **не всё приложение**, а
только *общий сервер данных* для команды (`GENSTUDIO_MODE=server`): проекты,
ассеты, пользователи. Идея автора — у каждого пользователя на своём ПК стоит
десктоп-приложение с ComfyUI и GPU, а Docker-сервер только хранит общие данные.

В этом режиме все «вычислительные» маршруты специально выключены и отвечают
`404` (см. `serverMode.js`): `/api/comfyui`, `/api/setup` (мастер установки),
`/api/settings`, `/api/meshes` и т. д. Поэтому окно «укажите путь к ComfyUI»
появляется, но кнопка **Next** ничего не делает — конфиг мастера не загружается.

## Что добавлено в этом форке

| Файл | Зачем |
| --- | --- |
| `Dockerfile`, стадия `standalone` | Образ со всем приложением в локальном режиме (как десктоп, но без Electron) |
| `docker-compose.standalone.yml` | Запуск этого образа рядом с вашим ComfyUI |
| `.env.standalone.example` | Шаблон настроек |
| `server.js` | Переменные `GENSTUDIO_COMFYUI_*` (адрес и папка ComfyUI) и `GENSTUDIO_BASIC_AUTH` (пароль на вход) |

Штатный `docker-compose.yml` не изменился и работает как раньше.

## Как это устроено

```
 браузер ──► genstudio (порт 3001)
                │  HTTP + WebSocket ─────────► ComfyUI-контейнер (порт 8188)
                │
                └─ /comfyui  ◄── общая папка на хосте ──►  папка ComfyUI в его контейнере
                   (сюда мастер качает модели)            (ComfyUI их видит)
```

* С ComfyUI приложение общается только по сети (`/prompt`, `/upload/image`,
  `/history`, `/view`, WebSocket), поэтому ему нужен **адрес** ComfyUI.
* Путь к папке нужен **только мастеру установки**: он скачивает модели в
  `<папка ComfyUI>/models/...`. Поэтому ту же папку, что смонтирована в ComfyUI,
  надо смонтировать и в контейнер 3D Gen Studio (в `/comfyui`).

## Пошагово

### 1. Узнайте две вещи о своём ComfyUI-контейнере

```bash
docker ps                                  # имя контейнера ComfyUI
docker inspect <comfyui> --format '{{json .Mounts}}' | jq   # где на хосте лежит папка с models/
docker inspect <comfyui> --format '{{json .NetworkSettings.Networks}}' | jq  # сеть
```

* **Папка на хосте**, в которой лежит `models/` (и обычно `custom_nodes/`), —
  это будет `COMFYUI_DIR`. Если модели смонтированы отдельным томом, см. раздел
  «Модели в отдельной папке» ниже.
* ComfyUI должен быть запущен с `--listen 0.0.0.0` (иначе другие контейнеры
  до него не достучатся).

### 2. Настройте `.env.standalone`

```bash
cp .env.standalone.example .env.standalone
nano .env.standalone
```

Минимум:

```ini
COMFYUI_DIR=/opt/comfyui                  # папка ComfyUI на хосте
GENSTUDIO_COMFYUI_URL=http://host.docker.internal
GENSTUDIO_COMFYUI_PORT=8188
GENSTUDIO_BASIC_AUTH=admin:сложный-пароль
```

**Вариант с общей docker-сетью** (если ComfyUI не публикует порт на хост):
раскомментируйте блоки `networks:` в `docker-compose.standalone.yml` и задайте

```ini
COMFYUI_NETWORK=имя_сети_comfyui          # из docker inspect выше
GENSTUDIO_COMFYUI_URL=http://имя_контейнера_comfyui
```

### 3. Запуск

```bash
docker compose -f docker-compose.standalone.yml --env-file .env.standalone up -d --build
docker compose -f docker-compose.standalone.yml --env-file .env.standalone logs -f
```

В логе должно быть:

```
🔧 ComfyUI settings from environment: {"url":"http://host.docker.internal","port":"8188","path":"/comfyui"}
🚀 3D Gen Studio Backend running at http://localhost:3001
```

Откройте `http://<ваш-сервер>:3001`.

### 4. Мастер установки (Setup Wizard)

1. **ComfyUI Folder** уже заполнено значением `/comfyui` — это путь *внутри
   контейнера*, менять не нужно. Кнопка **Browse** в Docker не работает (она
   только для Windows) — просто жмите **Next**.
2. **Models** — выберите нужные модели и качество (F16/Q8/Q4…). Ориентируйтесь
   на объём VRAM вашей видеокарты.
3. **Download** — модели скачиваются прямо в папку ComfyUI на хосте. Это десятки
   гигабайт; прогресс виден в окне.
4. **Workflows** — установка готовых workflow в библиотеку приложения.

После этого перезапустите ComfyUI или нажмите в нём «Refresh», чтобы он увидел
новые файлы моделей.

> **Важно: custom nodes.** Мастер качает только *модели*. Workflow используют
> кастомные ноды (ComfyUI-GGUF, Trellis2, Hunyuan3D, RMBG, QwenVL и др.).
> Их нужно поставить в ваш ComfyUI самостоятельно (например через ComfyUI-Manager →
> «Install Missing Custom Nodes», открыв нужный workflow из `setup/` в ComfyUI).
> Если нода не установлена, ComfyUI вернёт ошибку при запуске генерации.

### 5. Проверка связи с ComfyUI

```bash
docker compose -f docker-compose.standalone.yml exec genstudio \
  node -e "fetch(process.env.GENSTUDIO_COMFYUI_URL+':'+process.env.GENSTUDIO_COMFYUI_PORT+'/system_stats').then(r=>r.text()).then(console.log)"
```

Должен вернуться JSON с информацией о GPU. Если ошибка — проблема в адресе/сети
(см. шаг 1–2).

## Модели в отдельной папке

Если у ComfyUI модели смонтированы отдельно (например `/data/models`), добавьте
второй том в `docker-compose.standalone.yml`:

```yaml
    volumes:
      - genstudio-standalone-data:/app/data
      - ${COMFYUI_DIR}:/comfyui
      - /data/models:/comfyui-models
```

и в `.env.standalone`:

```ini
GENSTUDIO_COMFYUI_MODELS_PATH=/comfyui-models
```

## Права на файлы

По умолчанию контейнер работает от root, чтобы гарантированно писать в папку
ComfyUI. Если ComfyUI работает от другого пользователя и вы не хотите файлы с
владельцем root, задайте `GENSTUDIO_UID=1000:1000` (uid:gid владельца папки).

## Что не работает в Docker-режиме

Python-сервисы (mesh-tools :8200, SkinTokens :8300, Kimodo :8400,
MoCapAnything :8401) в образ не входят: это функции Auto UV, Auto Retopo,
Auto Rig, генерация анимации и превью-миниатюры мешей. Всё, что идёт через
ComfyUI и внешние API (Tripo, Hitem3D, Tencent и т.д.), работает. При желании
эти сервисы можно запустить отдельно (на машине с GPU, `python-server/run.sh`
и т.п.) и указать их адреса в Settings.

## Безопасность

В локальном режиме нет учётных записей: любой, кто достучался до порта, видит
Settings (включая API-ключи) и может запускать задачи. Поэтому:

* задайте `GENSTUDIO_BASIC_AUTH=логин:пароль`, и/или
* публикуйте порт только в локальную сеть / за reverse proxy с HTTPS
  (тогда `PUBLIC_BASE_URL=https://...` и `TRUST_PROXY_HEADERS=1`).

## Данные и бэкап

Все проекты, ассеты и база SQLite лежат в томе `genstudio-standalone-data`
(`/app/data` в контейнере):

```bash
docker run --rm -v 3dgenstudio_genstudio-standalone-data:/data -v "$PWD":/backup alpine \
  tar czf /backup/genstudio-data.tgz -C /data .
```

## Обновление из оригинального репозитория

```bash
git remote add upstream https://github.com/visualbruno/3DGenStudio.git   # один раз
git fetch upstream
git merge upstream/main
docker compose -f docker-compose.standalone.yml --env-file .env.standalone up -d --build
```
