#!/usr/bin/env bash
# Runs the whole stack with plain `docker` commands, for machines without the compose plugin.
#   scripts/stack.sh up     build and start PostGIS, the API and the web app
#   scripts/stack.sh db     start only PostGIS (for local development and tests)
#   scripts/stack.sh down   stop and remove the containers (volumes are kept)
#   scripts/stack.sh logs   follow the API log
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi

WEB_PORT="${CURBCUT_WEB_PORT:-4109}"
API_PORT="${CURBCUT_API_PORT:-8109}"
DB_PORT="${CURBCUT_DB_PORT:-5509}"
DB_PASSWORD="${CURBCUT_DB_PASSWORD:-curbcut}"
POSTGIS_IMAGE="${POSTGIS_IMAGE:-imresamu/postgis:17-3.5}"
MAP_STYLE="${VITE_MAP_STYLE_URL:-https://tiles.openfreemap.org/styles/positron}"
NET=curbcut-net

port_free() {
  if lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $1 is already in use. Set $2 to another port." >&2
    exit 1
  fi
}

running() { [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null || true)" = "true" ]; }

start_db() {
  docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null
  if running curbcut-db; then return; fi
  docker rm -f curbcut-db >/dev/null 2>&1 || true
  port_free "$DB_PORT" CURBCUT_DB_PORT
  docker run -d --name curbcut-db --network "$NET" \
    -e POSTGRES_USER=curbcut -e POSTGRES_PASSWORD="$DB_PASSWORD" -e POSTGRES_DB=curbcut \
    -p "127.0.0.1:$DB_PORT:5432" -v curbcut-db:/var/lib/postgresql/data "$POSTGIS_IMAGE" >/dev/null
  printf 'waiting for PostGIS'
  for _ in $(seq 1 60); do
    if docker exec curbcut-db pg_isready -U curbcut -d curbcut >/dev/null 2>&1 &&
       docker exec curbcut-db psql -U curbcut -d curbcut -c 'select 1' >/dev/null 2>&1; then
      echo ' ready'; return
    fi
    printf '.'; sleep 1
  done
  echo ' timed out' >&2; exit 1
}

case "${1:-up}" in
  db)
    start_db
    echo "PostGIS on localhost:$DB_PORT"
    ;;
  up)
    start_db
    docker build -q -t curbcut-api -f api/Dockerfile . >/dev/null
    docker build -q -t curbcut-web -f web/Dockerfile --build-arg VITE_MAP_STYLE_URL="$MAP_STYLE" . >/dev/null
    docker rm -f curbcut-api curbcut-web >/dev/null 2>&1 || true
    port_free "$API_PORT" CURBCUT_API_PORT
    port_free "$WEB_PORT" CURBCUT_WEB_PORT
    docker run -d --name curbcut-api --network "$NET" --network-alias api \
      -e CURBCUT_DATABASE_URL="postgresql://curbcut:$DB_PASSWORD@curbcut-db:5432/curbcut" \
      -e CURBCUT_SEED_DEMO="${CURBCUT_SEED_DEMO:-true}" \
      -e CURBCUT_REVIEWER_TOKEN="${CURBCUT_REVIEWER_TOKEN:-}" \
      -e CURBCUT_DEVICE_SALT="${CURBCUT_DEVICE_SALT:-curbcut-dev-salt}" \
      -e CURBCUT_CORS_ORIGINS="http://localhost:$WEB_PORT,http://127.0.0.1:$WEB_PORT" \
      -e CURBCUT_UPLOAD_DIR=/data/uploads \
      -p "127.0.0.1:$API_PORT:8109" -v curbcut-uploads:/data/uploads curbcut-api >/dev/null
    docker run -d --name curbcut-web --network "$NET" -p "127.0.0.1:$WEB_PORT:80" curbcut-web >/dev/null
    printf 'waiting for the API'
    for _ in $(seq 1 60); do
      if curl -fsS "http://localhost:$API_PORT/api/health" >/dev/null 2>&1; then echo ' ready'; break; fi
      printf '.'; sleep 1
    done
    echo "Curbcut is at http://localhost:$WEB_PORT  (API http://localhost:$API_PORT/api/health)"
    ;;
  down)
    docker rm -f curbcut-web curbcut-api curbcut-db >/dev/null 2>&1 || true
    docker network rm "$NET" >/dev/null 2>&1 || true
    echo "stopped. Data volumes curbcut-db and curbcut-uploads are kept (docker volume rm to delete)."
    ;;
  logs)
    docker logs -f curbcut-api
    ;;
  *)
    echo "usage: scripts/stack.sh up|db|down|logs" >&2; exit 2
    ;;
esac
