.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  .venv            Create a virtual environment"
	@echo "  clean            Remove the virtual environment"
	@echo "  pre-commit       Run pre-commit checks"
	@echo "  update-requirements Update the requirements.txt and requirements-dev.txt files"
	@echo "  requirements.txt  Generate the requirements.txt file"
	@echo "  requirements-dev.txt  Generate the requirements-dev.txt file"

GREEN := \033[0;32m
NC := \033[0m
RED := \033[0;31m
YELLOW := \033[0;33m

.venv:
	@./bin/install-micromamba-python-version.sh 3.11.0 .venv


	@echo "Installing dependencies from requirements-*.txt"
	@./.venv/bin/pip install -q pip-tools
	@./.venv/bin/pip-sync -q --pip-args "--no-deps" requirements*.txt

	@echo "[${GREEN} SUCCESS ${NC}] Created virtual environment with dependencies"

	@./.venv/bin/pip install -q ipykernel

	@echo "[${GREEN} SUCCESS ${NC}] Installed ipykernel for VSCode integration"


clean:
	@rm -rf .venv
	@echo "[${GREEN} SUCCESS ${NC}] Removed virtual environment"

pre-commit:
	pre-commit run --all-files

	@echo "[${GREEN} SUCCESS ${NC}] Pre-commit checks passed"

update-requirements:
	@$(MAKE) -B requirements.txt
	@$(MAKE) -B requirements-dev.txt

	@echo "[${GREEN} SUCCESS ${NC}] Updated requirements.txt and requirements-dev.txt"

requirements.txt:
	@./bin/install-micromamba-python-version.sh 3.11.0 .venv
	@./.venv/bin/pip install -q pip-tools
	@./.venv/bin/pip-compile -q --strip-extras

	@echo "[${GREEN} SUCCESS ${NC}] Updated requirements.txt"


my-min-devkit:
	@git clone https://github.com/Cycling74/min-devkit.git --recursive my-min-devkit


requirements-dev.txt: .venv
	@./bin/install-micromamba-python-version.sh 3.11.0 .venv
	@./.venv/bin/pip install -q pip-tools
	@./.venv/bin/pip-compile -q requirements-dev.in --strip-extras

	@echo "[${GREEN} SUCCESS ${NC}] Updated requirements-dev.txt"

max-sdk:
	@git clone https://github.com/Cycling74/max-sdk.git --recursive


string_synth:
	ruby my-min-devkit/script/create_package.rb string_synth

string_synth/libtorch:
	./bin/install-wget.sh
	mkdir -p string_synth
	wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.4.1.zip -O string_synth/libtorch.zip
	unzip string_synth/libtorch.zip -d string_synth
	rm string_synth/libtorch.zip
