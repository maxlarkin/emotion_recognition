#!/usr/bin/env sh
# Checks that each compose command starts exactly one backend (AGENTS.md, story 1.1):
#   default (COMPOSE_PROFILES=cpu from .env) -> backend + frontend
#   --profile gpu                             -> backend-gpu + frontend
set -eu
cd "$(dirname "$0")/.."

check() {
  expected="$1"; shift
  actual=$(docker compose "$@" config --services | sort | tr '\n' ' ' | sed 's/ $//')
  if [ "$actual" != "$expected" ]; then
    echo "FAIL: docker compose $* -> '$actual', expected '$expected'" >&2
    exit 1
  fi
  echo "ok: docker compose $* -> $actual"
}

check "backend frontend"
check "backend-gpu frontend" --profile gpu
