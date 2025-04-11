#!/usr/bin/bash

# =======================
# ===  Color Loggers  ===
# =======================
GREEN='\033[1;32m'; YELLOW='\033[1;33m'; RED='\033[1;31m'; BLUE='\033[1;34m'; CYAN='\033[1;36m'; NC='\033[0m'

log_info()   { echo -e "${GREEN}[INFO] $1${NC}"; }
log_warn()   { echo -e "${YELLOW}[WARN] $1${NC}"; }
log_error()  { echo -e "${RED}[ERROR] $1${NC}"; }
log_help()   { echo -e "${CYAN}$1${NC}"; }

# =========================
# === Sourced-Only Check ==
# =========================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    log_error "Please source this script instead of running it directly:"
    echo "    source ./scripts/setup.sh"
    exit 1
fi

# ======================
# === Project Paths  ===
# ======================
ROOT="$( cd "$( dirname "${BASH_SOURCE[1]}" )/.." && pwd )"
PROJECT_ROOT="$ROOT/NLPinitiative"

# ================================
# ===   Python Version Check   ===
# ================================
check_python_version() {
    REQUIRED="3.7"
    if ! command -v python3 &> /dev/null; then
        log_warn "python3 not found. Python >= 3.7 is recommended if you need Python tasks."
    else
        local PY_VER
        PY_VER=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if [[ $(printf '%s\n' "$REQUIRED" "$PY_VER" | sort -V | head -n1) != "$REQUIRED" ]]; then
            log_warn "Detected Python version: $PY_VER (older than $REQUIRED). Consider upgrading."
        else
            log_info "Detected Python version: $PY_VER"
        fi
    fi
}

# =================================
# ===  pipenv Check (No Prompt) ===
# =================================
check_pipenv() {
    if ! command -v pipenv &> /dev/null; then
        log_warn "pipenv not found. Install via 'pip install pipenv'."
    else
        log_info "Detected pipenv."
    fi
}

# =======================================
# ===   Virtual Env Existence Check   ===
# =======================================
check_virtualenv() {
    cd "$PROJECT_ROOT" || return 1

    if [ -d ./.venv ]; then
        log_info "Local .venv found. Activating venv..."
        source .venv/Scripts/activate
        log_info "Virtual Environment Activated."
    else
        log_warn "No local .venv found. Run 'build' to create and activate a local venv."
    fi
}

# ============================
# ===    Build Function    ===
# ============================
build() {
    cd "$PROJECT_ROOT" || return 1
    log_info "Running 'build': Cleaning project and then rebuilding virtual environment..."

    clean 

    log_info "Setting up new Virtual Environment..."
    python -m venv .venv

    log_info "Activating Virtual Environment..."
    source .venv/Scripts/activate

    log_info "Loading dependencies..."
    pip install pipenv
    pipenv install

    log_info "Build Complete."
}

# ========================
# ===  Clean Function  ===
# ========================
clean() {
    cd "$PROJECT_ROOT" || return 1
    log_info "Running 'clean': Removing .venv and any python cache and compiled binary files."

    if [ -n "$VIRTUAL_ENV" ]; then
        log_info "Deactivating Virtual Environment..."
        deactivate
        log_info "Virtual Environment Deactivated."
    fi

    if [ -d ./.venv ]; then
        log_info "Removing .venv directory..."
        rm -rf ./.venv
        log_info "Removed .venv directory."
    fi

    log_info "Removing cache and compiled binary files..."
    find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
    log_info "Cache and compiled binary files removed."

    log_info "Clean Complete."
}

# ============================
# === Generate Docs Command ==
# ============================
docs() {
    cd "$PROJECT_ROOT" || return 1
    case $1 in
        build)
            log_info "Building documentation..."
            mkdocs build
            log_info "Documentation built."
            ;;
        serve)
            log_info "Serving documentation..."
            mkdocs serve
            ;;
        deploy)
            log_info "Deploying documentation to GH Pages..."
            mkdocs gh-deploy --force
            log_info "Documentation deployed to GH Pages."
            ;;
        *)
            log_error "Specify 'build', 'serve' or 'deploy'. For example: docs build"
            ;;
    esac
}

# ============================
# ===    Config Settings   ===
# ============================
set() {
    cd "$PROJECT_ROOT" || return 1

    if [[ $# -lt 2 ]]; then
        log_error "Set Command Requires Two Arguments."
    else
        case $1 in
            bin_repo)
                python ./nlpinitiative/config.py -br "$2"
                ;;
            ml_repo)
                python ./nlpinitiative/config.py -mr "$2"
                ;;
            ds_repo)
                python ./nlpinitiative/config.py -dr "$2"
                ;;
            streamlit_repo)
                python ./nlpinitiative/config.py -sr "$2"
                ;;
            space_url)
                python ./nlpinitiative/config.py -su "$2"
                ;;
            model_url)
                python ./nlpinitiative/config.py -mu "$2"
                ;;
            dataset_url)
                python ./nlpinitiative/config.py -du "$2"
                ;;
            hf_token)
                python ./nlpinitiative/config.py -ht "$2"
                ;;
            *)
                log_error "Invalid set option."
                echo "Available 'set' options:"
                echo "  bin_repo <repo ID>          - Sets the binary model's repo ID in the pyproject.toml file."
                echo "  ml_repo <repo ID>           - Sets the multilabel model's repo ID in the pyproject.toml file."
                echo "  ds_repo <repo ID>           - Sets the dataset repo ID in the pyproject.toml file."
                echo "  streamlit_repo <repo ID>    - Sets the Streamlit App repo ID in the pyproject.toml file."
                echo "  space_url <base URL>        - Sets the base URL for HF Spaces in the pyproject.toml file."
                echo "  model_url <base URL>        - Sets the base URL for HF Model Repos in the pyproject.toml file."
                echo "  dataset_url <base URL>      - Sets the base URL for HF Dataset Repos in the pyproject.toml file."
                ;;
        esac
    fi
}

# ============================
# ===      Run  Tests      ===
# ============================
run() {
    cd "$PROJECT_ROOT" || return 1
    case $1 in
        tests)
            log_info "Running tests..."
            coverage run -m unittest discover
            coverage report -m
            coverage html
            log_info "Tests complete. Check coverage report in htmlcov/index.html"
            ;;
        ds_sync)
            log_info "Syncing dataset data..."
            python ./nlpinitiative/data_preparation/data_management.py
            ;;
        model_sync)
            log_info "Syncing model data..."
            python ./nlpinitiative/modeling/train.py
            ;;
        *)
            log_error "Invalid run option."
            echo "Available 'run' options:"
            echo "  tests - Runs the test suite."
            echo "  ds_sync - Syncs the dataset data."
            echo "  model_sync - Syncs the model data."
            ;;
    esac
}

# ============================
# ===     Lint Command     ===
# ============================
lint() {
    cd "$PROJECT_ROOT" || return 1

    log_info "Linting project..."
    flake8 nlpinitiative
	isort --check --diff --profile black nlpinitiative
	black --check --config pyproject.toml nlpinitiative
}

# ============================
# ===    Format Command    ===
# ============================
format() {
    cd "$PROJECT_ROOT" || return 1

    log_info "Fixing project formatting..."
    black --config pyproject.toml nlpinitiative
}

# ============================
# ===     Help Command     ===
# ============================
help() {
    cd "$PROJECT_ROOT" || return 1

    log_help "Usage: source ./scripts/setup.sh"
    echo "Available commands:"
    echo "  Miscellaneous commands:"
    echo "      help        - Show this help message."
    echo "==========================================="
    echo "  Project Building, Cleaning, etc. commands:"
    echo "      build       - Cleans and reinstalls Python dependencies."
    echo "      clean       - Cleans project (i.e., deactivates venv, removes .venv and clears project of python cache and binary files)."
    echo "      lint        - Lints the project files."
    echo "      format      - Corrects the files identified while linting."
    echo "      run tests   - Runs the test suite."
    echo "      run ds_sync - Syncs the dataset data."
    echo "      run model_sync - Syncs the model data."
    echo "==========================================="
    echo "  'docs' command options:"
    echo "      docs build      - Generates mkdocs for the project."
    echo "      docs serve      - Serves documentation locally."
    echo "      docs deploy     - Deploys documentation to associated GH Pages."
    echo "==========================================="
    echo "  'set' command options:"
    echo "      set bin_repo <repo ID>          - Sets the binary model's repo ID in the pyproject.toml file."
    echo "      set ml_repo <repo ID>           - Sets the multilabel model's repo ID in the pyproject.toml file."
    echo "      set ds_repo <repo ID>           - Sets the dataset repo ID in the pyproject.toml file."
    echo "      set streamlit_repo <repo ID>    - Sets the Streamlit App repo ID in the pyproject.toml file."
    echo "      set space_url <base URL>        - Sets the base URL for HF Spaces in the pyproject.toml file."
    echo "      set model_url <base URL>        - Sets the base URL for HF Model Repos in the pyproject.toml file."
    echo "      set dataset_url <base URL>      - Sets the base URL for HF Dataset Repos in the pyproject.toml file."
    echo "      set hf_token <HF Token>         - Sets the HF personal token in the pyproject.toml file."
}

log_info "Loading setup.sh script..."

check_python_version
check_pipenv
check_virtualenv

log_info "setup.sh loaded. Type 'help' for usage."
log_info "Try 'build' then 'run dev' to get started."