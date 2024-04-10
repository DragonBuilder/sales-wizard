#!/bin/bash

module=$1

if [[ -z "$module" ]]; then
    echo "no agent module name provided"
    echo "usage: ./run_agent.sh <module_name>"
    exit 1
fi

export $(xargs < .env)
poetry run python -m sales_wizard.$module