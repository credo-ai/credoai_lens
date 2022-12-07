#!/usr/bin/env bash

set -e
set -x

PYTHONPATH=credoai pytest --cov-config=.coveragerc --cov=credoai --cov-report=term-missing tests "${@}"
