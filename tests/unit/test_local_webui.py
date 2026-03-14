from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from karaoke_gen.local_webui import create_app


def test_index_page_has_audio_and_lyrics_upload_inputs(tmp_path):
    client = TestClient(create_app(base_dir=tmp_path))

    response = client.get("/")

    assert response.status_code == 200
    assert 'name="audio_file"' in response.text
    assert 'name="lyrics_file"' in response.text


def test_create_job_uses_offline_cli_command_and_env(tmp_path):
    client = TestClient(create_app(base_dir=tmp_path))
    process = MagicMock()
    process.poll.return_value = None

    with patch("karaoke_gen.local_webui.subprocess.Popen", return_value=process) as mock_popen:
        response = client.post(
            "/api/jobs",
            data={
                "artist": "ABBA",
                "title": "Waterloo",
                "whisper_model_size": "small",
                "whisper_device": "cpu",
            },
            files={
                "audio_file": ("waterloo.flac", b"audio-data", "audio/flac"),
                "lyrics_file": ("waterloo.txt", b"my lyrics", "text/plain"),
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "running"

    command = mock_popen.call_args.args[0]
    env = mock_popen.call_args.kwargs["env"]

    assert "--offline" in command
    assert "--skip_transcription_review" in command
    assert "--skip_instrumental_review" in command
    assert "--lyrics_file" in command
    assert env["SKIP_CORRECTION"] == "false"
    assert env["USE_AGENTIC_AI"] == "0"
    assert env["WHISPER_MODEL_SIZE"] == "small"
    assert env["WHISPER_DEVICE"] == "cpu"
    assert "AUDIOSHAKE_API_TOKEN" not in env
    assert "RUNPOD_API_KEY" not in env
