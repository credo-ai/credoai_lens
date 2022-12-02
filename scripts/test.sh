#!/usr/bin/env bash

set -e
set -x

PYTHONPATH=credoai pytest \
  --junitxml=junit.xml \
  --cov-report=term-missing \
  --cov=credoai tests/ "${@}"
