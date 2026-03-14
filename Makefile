# Makefile for karaoke-gen project
# Run `make help` to see available commands

.PHONY: help install install-backend install-frontend build-frontend dev-install test test-unit test-backend test-e2e test-frontend test-all lint clean emulators-start emulators-stop

# Per-stage timeout in seconds (default: 10 minutes)
# Prevents hangs from stale dev servers, broken emulators, etc.
# Override with: make test TEST_TIMEOUT=300
TEST_TIMEOUT ?= 600
# Full frontend test runs include the serial Playwright regression suite,
# which can exceed the general 10 minute cap in local CI-like environments.
FRONTEND_TEST_TIMEOUT ?= 1200

# Default target
help:
	@echo "Available commands:"
	@echo "  make test           - Run ALL tests (backend + frontend, installs deps automatically)"
	@echo "  make test-backend   - Run backend tests only (unit + emulator)"
	@echo "  make test-frontend  - Run frontend tests only (unit + E2E)"
	@echo "  make test-unit      - Run unit tests only (karaoke_gen package)"
	@echo "  make test-e2e       - Run emulator tests with auto-start/stop"
	@echo "  make install        - Install all dependencies (backend + frontend)"
	@echo "  make build-frontend - Build frontend and copy to Python package"
	@echo "  make dev-install    - Build frontend + pip install -e . (for local CLI testing)"
	@echo "  make lint           - Run linter checks"
	@echo "  make emulators-start - Start GCP emulators for local development"
	@echo "  make emulators-stop  - Stop GCP emulators"
	@echo ""
	@echo "Before committing, run: make test"

# Install dependencies (only if needed)
install-backend:
	@if [ ! -d "$$(poetry env info --path 2>/dev/null)" ] || ! poetry run python -c "import fastapi" 2>/dev/null; then \
		echo "=== Installing backend dependencies ==="; \
		poetry install; \
	fi

install-frontend:
	@if [ ! -d "frontend/node_modules" ]; then \
		echo "=== Installing frontend dependencies ==="; \
		cd frontend && npm install; \
	fi

install: install-backend install-frontend

# Build frontend and copy to Python package
build-frontend: install-frontend
	@echo "=== Building frontend ==="
	cd frontend && npm run build
	@echo "=== Copying build to Python package ==="
	rm -rf karaoke_gen/nextjs_frontend/out
	mkdir -p karaoke_gen/nextjs_frontend/out
	cp -r frontend/out/* karaoke_gen/nextjs_frontend/out/
	@echo "✅ Frontend built and copied to karaoke_gen/nextjs_frontend/out/"

# Build frontend + install package in editable mode for local CLI testing
# After running this, `karaoke-gen` CLI will use the freshly built frontend.
dev-install: build-frontend
	@echo "=== Installing package in editable mode ==="
	pip install -e .
	@echo ""
	@echo "✅ Ready! You can now run karaoke-gen with the updated frontend."
	@echo "   Example: karaoke-gen input.flac \"Artist\" \"Title\""

# Run unit tests for karaoke_gen package
test-unit: install-backend
	@echo "=== Running karaoke_gen unit tests ==="
	timeout $(TEST_TIMEOUT) poetry run pytest tests/unit/ -v --cov=karaoke_gen --cov-report=term-missing --cov-fail-under=69

# Run backend unit tests (excludes emulator tests)
test-backend-unit: install-backend
	@echo "=== Running backend unit tests ==="
	timeout $(TEST_TIMEOUT) poetry run pytest backend/tests/ --ignore=backend/tests/emulator -v

# Run E2E integration tests with emulators
test-e2e: install-backend
	@echo "=== Running E2E integration tests with emulators ==="
	@TEST_TIMEOUT=$(TEST_TIMEOUT) ./scripts/run-emulator-tests.sh

# Run all backend tests (unit + emulator)
test-backend: test-unit test-backend-unit test-e2e
	@echo ""
	@echo "✅ Backend tests passed!"

# Run frontend tests (unit + E2E)
test-frontend: install-frontend
	@echo "=== Running frontend tests ==="
	cd frontend && timeout $(FRONTEND_TEST_TIMEOUT) npm run test:all

# Run ALL tests (backend + frontend) - use this before committing!
test: test-backend test-frontend
	@echo ""
	@echo "✅ All tests passed! Ready to commit."

# Alias for test
test-all: test

# Lint checks
lint:
	@echo "=== Running lint checks ==="
	poetry run ruff check karaoke_gen/ backend/
	poetry run ruff format --check karaoke_gen/ backend/

# Start emulators for local development
emulators-start:
	@echo "=== Starting GCP emulators ==="
	./scripts/start-emulators.sh

# Stop emulators
emulators-stop:
	@echo "=== Stopping GCP emulators ==="
	./scripts/stop-emulators.sh

# Clean up temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml 2>/dev/null || true
	@echo "Cleaned up temporary files"
