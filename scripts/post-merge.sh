#!/bin/bash
set -e
npm install
pip install -e . 2>/dev/null || uv sync 2>/dev/null || true
