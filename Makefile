PYTHON  := python3
VENV    := .venv
PIP     := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit
PY      := $(VENV)/bin/python

# ── install ────────────────────────────────────────────────────────────────────
# Creates a virtual environment (once) and installs all dependencies.
# Safe to re-run: the `if [ ! -d ]` guard skips venv creation if it exists.
.PHONY: install
install:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV); \
	fi
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip --quiet
	$(PIP) install -r requirements.txt --quiet
	@echo "Installation complete."

# ── run ────────────────────────────────────────────────────────────────────────
# Launches the Streamlit web app.
# Depends on `install` so a cold `make run` on a fresh clone works end-to-end.
.PHONY: run
run: install
	@echo "Starting BU Green Line Trip Calculator..."
	$(STREAMLIT) run scripts/app.py

# ── dataset ────────────────────────────────────────────────────────────────────
# (Optional) Rebuilds the ML-ready Parquet from raw MBTA CSVs + weather.
# Requires an internet connection; downloads ~24 months of data from Hugging Face.
# Pre-built artifacts are already committed — only run this to regenerate from scratch.
# .PHONY: dataset
# dataset: install
#	 @echo "Building dataset (downloads raw MBTA data from Hugging Face)..."
#	 $(PY) scripts/dataset_creation.py

# ── train ──────────────────────────────────────────────────────────────────────
# (Optional) Retrains XGBoost models and saves artifacts to scripts/model_artifacts/.
# Pre-trained artifacts are already committed — only run this after rebuilding the dataset.
.PHONY: train
train: install
	@echo "Training models and saving artifacts..."
	$(PY) scripts/model.py

# ── test ───────────────────────────────────────────────────────────────────────
# Runs the unit test suite.
.PHONY: test
test: install
	@echo "Running tests..."
	$(PY) -m pytest tests/ -v

# ── clean ──────────────────────────────────────────────────────────────────────
# Removes the virtual environment and any locally generated dataset files.
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	rm -f bu_green_line_gold.parquet
	@echo "Done."
