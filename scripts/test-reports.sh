#!/usr/bin/env bash

set -e
set -x

bash scripts/test.sh \
  --cov-report=xml:test-reports/cov.xml \
  --cov-report=html:test-reports/html \
  "${@}"
