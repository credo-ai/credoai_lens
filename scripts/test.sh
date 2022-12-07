#!/usr/bin/env bash

set -e
set -x

PYTHONPATH=credoai pytest \
  --cove-config=.coveragerc \
  --junitxml=junit.xml \
  --cov-report=term-missing \
  --cov=credoai tests/ "${@}"
