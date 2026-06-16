.PHONY: verify setup demo test lint format clean

# Day 0: run all bottleneck checks before touching any architecture
verify:
	uv run python scripts/day0_verify.py

# Full environment setup
setup:
	uv sync --extra dev
	python scripts/setup_ollama.py
	python scripts/download_weights.py
	@echo Setup complete. Run make verify to confirm everything works.

# Generate pitch template PNG
pitch-template:
	uv run python scripts/generate_pitch_template.py

# Launch Gradio demo
demo:
	uv run python app/gradio_app.py

# Run all tests
test:
	uv run pytest tests/ -v --tb=short

# Unit tests only (no weights required)
test-unit:
	uv run pytest tests/unit/ -v

# Integration tests (requires weights/ and a test clip)
test-integration:
	uv run pytest tests/integration/ -v

# Lint
lint:
	uv run ruff check gaffer/ app/ scripts/ tests/

# Format
format:
	uv run ruff format gaffer/ app/ scripts/ tests/

# Benchmark full pipeline
benchmark:
	uv run python scripts/benchmark.py --clip data/test_clips/sample.mp4 --output outputs/benchmark/

# Export YOLO model to OpenVINO
export-openvino:
	uv run python scripts/export_openvino.py

# Clean generated artifacts (keeps weights and data)
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf outputs/
