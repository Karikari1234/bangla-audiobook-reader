#!/usr/bin/env bash
# usage: ./run.sh [books_dir] [port]
cd "$(dirname "$0")"
BOOKS="${1:-$HOME/Books}"
mkdir -p "$BOOKS"
(sleep 3 && open "http://localhost:${2:-8766}") &
exec .venv/bin/python library_server.py "$BOOKS" "${2:-8766}"
