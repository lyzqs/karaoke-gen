#!/bin/bash
# Run integration tests with GCP emulators
# This script starts emulators, runs tests, and stops emulators

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_TIMEOUT="${TEST_TIMEOUT:-600}"

echo "🧪 Running integration tests with GCP emulators"
echo "=============================================="
echo ""

emulators_running() {
    curl -sf http://127.0.0.1:8080 >/dev/null 2>&1 && \
    curl -sf http://127.0.0.1:4443/storage/v1/b >/dev/null 2>&1
}

STARTED_EMULATORS=0

if emulators_running; then
    echo "Step 1/3: Reusing already-running emulators..."
elif docker info >/dev/null 2>&1; then
    echo "Step 1/3: Starting emulators..."
    "$SCRIPT_DIR/start-emulators.sh"
    STARTED_EMULATORS=1
else
    echo "⚠️  Skipping emulator integration tests: Docker is unavailable and emulators are not already running."
    exit 0
fi
echo ""

# Only stop emulators if this script started them.
if [ "$STARTED_EMULATORS" -eq 1 ]; then
    trap "$SCRIPT_DIR/stop-emulators.sh" EXIT
fi

# Set environment variables for tests
export FIRESTORE_EMULATOR_HOST="127.0.0.1:8080"
export STORAGE_EMULATOR_HOST="http://127.0.0.1:4443"
export GOOGLE_CLOUD_PROJECT="test-project"
export GCS_BUCKET_NAME="test-bucket"
export FIRESTORE_COLLECTION="jobs"
export ENVIRONMENT="test"
export ADMIN_TOKENS="test-admin-token"

# Determine how to run pytest
if command -v poetry &> /dev/null && [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
    echo "Step 2/3: Using poetry environment..."
    PYTEST_CMD="poetry run pytest"
elif [ -d "$PROJECT_ROOT/backend/venv" ]; then
    echo "Step 2/3: Activating Python virtual environment..."
    source "$PROJECT_ROOT/backend/venv/bin/activate"
    PYTEST_CMD="pytest"
else
    echo "⚠️  Warning: No poetry or backend/venv found. Using system Python."
    PYTEST_CMD="pytest"
fi

echo ""
echo "Step 3/3: Running emulator integration tests..."
echo ""
echo "Environment check:"
echo "  FIRESTORE_EMULATOR_HOST=$FIRESTORE_EMULATOR_HOST"
echo "  STORAGE_EMULATOR_HOST=$STORAGE_EMULATOR_HOST"
echo "  GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT"
echo ""

cd "$PROJECT_ROOT"

# Run the emulator integration tests with more verbose output on failure
if timeout "$TEST_TIMEOUT" $PYTEST_CMD backend/tests/emulator/ -v --tb=short --color=yes --log-cli-level=ERROR; then
    echo ""
    echo "✅ All emulator integration tests passed!"
    echo ""
    exit 0
else
    echo ""
    echo "❌ Some emulator integration tests failed"
    echo ""
    echo "Debugging info:"
    echo "Firestore emulator status:"
    curl -s http://127.0.0.1:8080 && echo "  Firestore responding" || echo "  Firestore NOT responding"
    echo "GCS emulator status:"
    curl -s http://127.0.0.1:4443/storage/v1/b && echo "  GCS responding" || echo "  GCS NOT responding"
    echo ""
    echo "Firestore emulator logs (last 30 lines):"
    tail -30 "$PROJECT_ROOT/.emulator-logs/firestore.log" 2>&1 || echo "  No logs found"
    echo ""
    exit 1
fi
