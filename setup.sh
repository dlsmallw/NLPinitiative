#!/usr/bin/bash

build() {
    clean 

    python -m venv .venv
    source .venv/Scripts/activate
    pip install pipenv
    pipenv install
}

clean() {
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi

    if [ -d ./.venv ]; then
        rm -rf ./.venv
    fi

    find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
}

docs() {
    case $1 in
        build)
            mkdocs build
            ;;
        serve)
            mkdocs serve
            ;;
        deploy)
            mkdocs gh-deploy --force
            ;;
        *)
            log_error "Specify 'build', 'serve' or 'deploy'. For example: docs build"
            ;;
    esac
}

lint() {
    flake8 nlpinitiative
	isort --check --diff --profile black nlpinitiative
	black --check --config pyproject.toml nlpinitiative
}

format() {
    black --config pyproject.toml nlpinitiative
}