import subprocess
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from karaoke_gen.local_webui import LocalJob, LocalWebUIServer, create_app


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


def test_refresh_job_exposes_playable_mp4_preview(tmp_path):
    server = LocalWebUIServer(base_dir=tmp_path)
    output_dir = tmp_path / "job" / "output"
    output_dir.mkdir(parents=True)
    video_path = output_dir / "finals" / "karaoke.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"video-bytes")

    process = MagicMock()
    process.poll.return_value = 0

    job = LocalJob(
        job_id="job123",
        artist="Test Artist",
        title="Test Title",
        created_at="2026-03-14T00:00:00+00:00",
        work_dir=tmp_path / "job",
        upload_dir=tmp_path / "job" / "uploads",
        output_dir=output_dir,
        log_path=tmp_path / "job" / "job.log",
        command=["python"],
        env_overrides={},
        process=process,
    )

    with patch("karaoke_gen.local_webui.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            stdout='{"streams":[{"codec_type":"video"}],"format":{"duration":"12.5"}}'
        )
        server._refresh_job(job)

    assert job.status == "completed"
    assert job.primary_video_url == "/api/jobs/job123/files/finals/karaoke.mp4"
    assert any(output["label"] == "finals/karaoke.mp4" for output in job.outputs)


def test_refresh_job_fails_when_mp4_is_not_playable(tmp_path):
    server = LocalWebUIServer(base_dir=tmp_path)
    output_dir = tmp_path / "job" / "output"
    output_dir.mkdir(parents=True)
    video_path = output_dir / "karaoke.mp4"
    video_path.write_bytes(b"broken-video")

    process = MagicMock()
    process.poll.return_value = 0

    job = LocalJob(
        job_id="job123",
        artist="Test Artist",
        title="Test Title",
        created_at="2026-03-14T00:00:00+00:00",
        work_dir=tmp_path / "job",
        upload_dir=tmp_path / "job" / "uploads",
        output_dir=output_dir,
        log_path=tmp_path / "job" / "job.log",
        command=["python"],
        env_overrides={},
        process=process,
    )

    with patch(
        "karaoke_gen.local_webui.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, ["ffprobe"]),
    ):
        server._refresh_job(job)

    assert job.status == "failed"
    assert job.primary_video_url is None
    assert "playable MP4" in job.error
