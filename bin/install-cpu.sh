#!/usr/bin/env bash
virtualenv -p /usr/bin/python3.6 .venv
source .venv/bin/activate
pip install -r ./requirements/requirements.txt
pip install -r ./requirements/requirements-cpu.txt