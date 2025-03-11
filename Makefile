GREEN := \033[0;32m
NC := \033[0m
RED := \033[0;31m
YELLOW := \033[0;33m

SHELL := /bin/bash
PYTHON_VERSION := $(shell cat .python-version)

# Find the research directory, i.e. the repo root
RESEARCH_DIR := $(shell git rev-parse --show-toplevel)

.venv: faust_python
	@echo "Creating virtual environment with Python $(PYTHON_VERSION)..."
	@$(RESEARCH_DIR)/bin/install-micromamba-python-version.sh $(PYTHON_VERSION) .venv


	@echo "Installing dependencies from requirements*.txt"
	@./.venv/bin/pip install -q pip-tools
	@./.venv/bin/pip-sync -q --pip-args "--no-deps" requirements*.txt

	@echo -e "[${GREEN} OK ${NC}] Created virtual environment with dependencies"

	@./.venv/bin/pip install -q ipykernel

	@./.venv/bin/pip install ./faust_python

	@echo -e "[${GREEN} OK ${NC}] Installed ipykernel for VSCode integration"


clean:
	@rm -rf .venv
	@echo -e "[${GREEN} OK ${NC}] Removed virtual environment"

pre-commit:
	pre-commit run --all-files

	@echo -e "[${GREEN} OK ${NC}] Pre-commit checks passed"

update-requirements:
	@$(MAKE) -B requirements.txt
	@$(MAKE) -B requirements-dev.txt

	@echo -e "[${GREEN} OK ${NC}] Updated requirements.txt and requirements-dev.txt"

requirements.txt: requirements.in
	@echo -e "[ 🦙 ] Creating requirements.txt"
	@./.venv/bin/pip-compile --upgrade requirements.in
	@if [ -f requirements-test.txt ]; then \
		./.venv/bin/pip-compile --upgrade requirements-test.in -o requirements-test.txt; \
	fi

	@echo -e "[${GREEN} OK ${NC}] Updated requirements.txt"

requirements-dev.txt: requirements-dev.in
	@echo -e "[ 🦙 ] Creating requirements-dev.txt"
	@./.venv/bin/pip-compile --upgrade requirements-dev.in

	@echo -e "[${GREEN} OK ${NC}] Updated requirements-dev.txt"

test:
	make requirements.txt requirements-dev.txt
	make .venv
	@echo -e "[ 🦙 ] Running tests"
	@./.venv/bin/python -m pytest
	@echo -e "[${GREEN} OK ${NC}] Tests passed"


faust_python:
	git clone https://github.com/hrtlacek/faust_python.git

	# remove instances of ", float128" from ./faust_python/FAUSTPy/python_dsp.py
	sed -i '' 's/, float128//g' ./faust_python/FAUSTPy/python_dsp.py
