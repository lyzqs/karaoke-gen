"""
Integration tests using local GCP emulators.

These tests require the emulator environment to be configured before pytest
starts. Run them via the emulator-specific helper script instead of the default
unit-test command.
"""
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests


def emulators_running() -> bool:
    """Check if GCP emulators are reachable."""
    try:
        requests.get("http://127.0.0.1:8080", timeout=1)
        requests.get("http://127.0.0.1:4443", timeout=1)
        return True
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False


EMULATOR_CONFIGURED = bool(os.environ.get("FIRESTORE_EMULATOR_HOST")) and bool(
    os.environ.get("STORAGE_EMULATOR_HOST")
)

# Skip all tests in this module unless the emulator environment is explicitly
# configured for this pytest invocation.
pytestmark = pytest.mark.skipif(
    not EMULATOR_CONFIGURED or not emulators_running(),
    reason="GCP emulators must be configured and running before pytest starts",
)

if EMULATOR_CONFIGURED and emulators_running():
    from fastapi.testclient import TestClient
    from backend.main import app
else:
    TestClient = None
    app = None


@pytest.fixture(scope="module", autouse=True)
def setup_gcs_bucket():
    """Create GCS bucket in emulator before tests."""
    try:
        response = requests.post(
            "http://localhost:4443/storage/v1/b",
            json={"name": os.environ.get("GCS_BUCKET_NAME", "test-bucket")},
            params={"project": os.environ.get("GOOGLE_CLOUD_PROJECT", "test-project")},
            timeout=5,
        )
        if response.status_code in [200, 409]:
            print("✅ GCS bucket ready")
    except Exception as exc:
        print(f"⚠️  GCS bucket setup failed: {exc}")
    yield


@pytest.fixture(scope="module")
def mock_worker_service():
    """Mock the worker service to prevent background tasks."""
    with patch("backend.api.routes.jobs.worker_service") as mock:
        mock.trigger_audio_worker = AsyncMock(return_value=True)
        mock.trigger_lyrics_worker = AsyncMock(return_value=True)
        mock.trigger_screens_worker = AsyncMock(return_value=True)
        mock.trigger_video_worker = AsyncMock(return_value=True)
        yield mock


@pytest.fixture(scope="module")
def mock_theme_service():
    """Mock theme service so emulator tests don't depend on GCS theme metadata."""
    service = MagicMock()
    service.get_default_theme_id.return_value = "nomad"
    service.theme_exists.return_value = True
    service.prepare_job_style.return_value = ("themes/nomad/style_params.json", {})
    service.get_youtube_description.return_value = None
    return service


@pytest.fixture(scope="module")
def client(mock_worker_service, mock_theme_service):
    """Create FastAPI test client with mocked workers."""
    with patch("backend.api.routes.file_upload.worker_service", mock_worker_service), \
         patch("backend.api.routes.jobs.get_theme_service", return_value=mock_theme_service), \
         patch("backend.api.routes.file_upload.get_theme_service", return_value=mock_theme_service):
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def auth_headers():
    """Auth headers for testing."""
    return {"Authorization": "Bearer test-admin-token"}


class TestEmulatorBasics:
    """Basic emulator connectivity tests."""

    def test_health_endpoint(self, client, auth_headers):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_root_endpoint(self, client, auth_headers):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["service"] == "karaoke-gen-backend"


class TestJobCreation:
    """Test job creation with Firestore emulator."""

    def test_create_job_simple(self, client, auth_headers):
        response = client.post(
            "/api/jobs",
            headers=auth_headers,
            json={"url": "https://youtube.com/watch?v=test123"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "job_id" in data

    def test_create_job_with_metadata(self, client, auth_headers):
        response = client.post(
            "/api/jobs",
            headers=auth_headers,
            json={
                "url": "https://youtube.com/watch?v=test",
                "artist": "Test Artist",
                "title": "Test Song",
            },
        )

        assert response.status_code == 200
        assert "job_id" in response.json()


class TestJobRetrieval:
    """Test job retrieval from Firestore emulator."""

    def test_create_and_get_job(self, client, auth_headers):
        create_resp = client.post(
            "/api/jobs",
            headers=auth_headers,
            json={
                "url": "https://youtube.com/watch?v=abc123",
                "artist": "Test Artist",
                "title": "Test Song",
            },
        )
        assert create_resp.status_code == 200
        job_id = create_resp.json()["job_id"]

        time.sleep(0.2)

        get_resp = client.get(f"/api/jobs/{job_id}", headers=auth_headers)
        assert get_resp.status_code == 200

        job = get_resp.json()
        assert job["job_id"] == job_id
        assert job["status"] == "pending"
        assert job["artist"] == "Test Artist"
        assert job["title"] == "Test Song"

    def test_get_nonexistent_job(self, client, auth_headers):
        response = client.get("/api/jobs/nonexistent-id", headers=auth_headers)
        assert response.status_code == 404


class TestJobList:
    """Test listing jobs from Firestore."""

    def test_list_jobs(self, client, auth_headers):
        for i in range(3):
            client.post(
                "/api/jobs",
                headers=auth_headers,
                json={"url": f"https://youtube.com/watch?v=list{i}"},
            )

        time.sleep(0.2)

        response = client.get("/api/jobs", headers=auth_headers)
        assert response.status_code == 200

        jobs = response.json()
        assert isinstance(jobs, list)
        assert len(jobs) >= 3


class TestJobDeletion:
    """Test job deletion."""

    def test_delete_job(self, client, auth_headers):
        create_resp = client.post(
            "/api/jobs",
            headers=auth_headers,
            json={"url": "https://youtube.com/watch?v=delete-me"},
        )
        job_id = create_resp.json()["job_id"]

        time.sleep(0.2)

        del_resp = client.delete(f"/api/jobs/{job_id}", headers=auth_headers)
        assert del_resp.status_code == 200

        get_resp = client.get(f"/api/jobs/{job_id}", headers=auth_headers)
        assert get_resp.status_code == 404


class TestJobUpdates:
    """Test job status updates."""

    def test_cancel_job(self, client, auth_headers):
        create_resp = client.post(
            "/api/jobs",
            headers=auth_headers,
            json={"url": "https://youtube.com/watch?v=cancel-me"},
        )
        job_id = create_resp.json()["job_id"]

        time.sleep(0.2)

        cancel_resp = client.post(
            f"/api/jobs/{job_id}/cancel",
            headers=auth_headers,
            json={"reason": "test cancellation"},
        )
        assert cancel_resp.status_code == 200

        get_resp = client.get(f"/api/jobs/{job_id}", headers=auth_headers)
        assert get_resp.status_code == 200
        assert get_resp.json()["status"] == "cancelled"


class TestFileUpload:
    """Test file upload with GCS emulator."""

    def test_upload_file(self, client, auth_headers):
        response = client.post(
            "/api/jobs/upload",
            headers={"Authorization": auth_headers["Authorization"]},
            files={"file": ("test.flac", b"fake audio data for testing", "audio/flac")},
            data={"artist": "Upload Artist", "title": "Upload Song"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "job_id" in data

        job_id = data["job_id"]
        time.sleep(0.2)

        get_resp = client.get(f"/api/jobs/{job_id}", headers=auth_headers)
        assert get_resp.status_code == 200
        job = get_resp.json()
        assert job["artist"] == "Upload Artist"
        assert job["title"] == "Upload Song"
        assert "input_media_gcs_path" in job


class TestInternalEndpoints:
    """Test internal worker endpoints."""

    def test_internal_workers_exist(self, client, auth_headers):
        create_resp = client.post(
            "/api/jobs",
            headers=auth_headers,
            json={"url": "https://youtube.com/watch?v=worker-test"},
        )
        job_id = create_resp.json()["job_id"]

        time.sleep(0.2)

        response = client.post(
            "/api/internal/workers/audio",
            headers=auth_headers,
            json={"job_id": job_id},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "started"


print("✅ Emulator integration tests ready to run")
