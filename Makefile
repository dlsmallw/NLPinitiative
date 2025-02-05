#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = NLPInitiative
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	pipenv install

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 nlpinitiative
	isort --check --diff --profile black nlpinitiative
	black --check --config pyproject.toml nlpinitiative

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml nlpinitiative

## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	pipenv --python $(PYTHON_VERSION)

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
## Import Raw Data from source
.PHONY: data_import
data_import: requirements
	$(PYTHON_INTERPRETER) nlpinitiative/data_preparation/data_import.py $(FLAG_ARG) $(FILEPATH_ARG)


## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) nlpinitiative/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
