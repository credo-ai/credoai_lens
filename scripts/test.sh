#!/usr/bin/env bash

set -e
set -x

PYTHONPATH=credoai pytest --cov=credoai --cov-report=term-missing tests "${@}"
