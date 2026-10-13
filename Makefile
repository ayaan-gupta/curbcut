# Curbcut. `make up` is the one command: PostGIS, the API and the web app in Docker.
PY := .venv/bin/python
TEST_DB := postgresql://curbcut:$${CURBCUT_DB_PASSWORD:-curbcut}@localhost:$${CURBCUT_DB_PORT:-5509}/curbcut_test
DEV_DB := postgresql://curbcut:$${CURBCUT_DB_PASSWORD:-curbcut}@localhost:$${CURBCUT_DB_PORT:-5509}/curbcut

.PHONY: up down logs setup dev test test-nodb demo

up:            ## build and run everything in Docker: web on 4109, API on 8109, PostGIS on 5509
	scripts/stack.sh up

down:          ## stop the containers (data volumes are kept)
	scripts/stack.sh down

logs:
	scripts/stack.sh logs

setup:         ## local toolchain for development and tests: a venv and node_modules
	test -d .venv || /opt/homebrew/bin/python3.12 -m venv .venv 2>/dev/null || python3.12 -m venv .venv
	.venv/bin/pip install -q --upgrade pip
	.venv/bin/pip install -q -r requirements-dev.txt
	cd web && npm ci --no-audit --no-fund

dev: setup     ## PostGIS in Docker, the API with reload and the Vite dev server, both on loopback
	scripts/stack.sh db
	(cd api && CURBCUT_DATABASE_URL=$(DEV_DB) ../.venv/bin/uvicorn curbcut.main:app --reload --host 127.0.0.1 --port $${CURBCUT_API_PORT:-8109}) & \
	(cd web && npm run dev) ; kill %1

test: setup    ## every test, with the store-backed ones run on both the memory store and PostGIS
	scripts/stack.sh db
	CURBCUT_TEST_DATABASE_URL=$(TEST_DB) $(PY) -m pytest -q
	cd web && npm run typecheck && npm test

test-nodb: setup  ## the same without Docker: the PostGIS half of the store tests is skipped
	$(PY) -m pytest -q
	cd web && npm run typecheck && npm test

demo:          ## record docs/demo.gif against a running stack
	cd web && npm run demo
