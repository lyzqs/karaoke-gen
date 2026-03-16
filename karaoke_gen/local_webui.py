import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from karaoke_gen.utils import sanitize_filename

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    _FASTAPI_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - dependency availability is environment-specific
    FastAPI = None  # type: ignore[assignment]
    File = None  # type: ignore[assignment]
    Form = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[assignment]
    UploadFile = None  # type: ignore[assignment]
    FileResponse = None  # type: ignore[assignment]
    HTMLResponse = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]
    _FASTAPI_IMPORT_ERROR = exc


OFFLINE_DISABLED_ENV_VARS = [
    "AUDIOSHAKE_API_TOKEN",
    "RUNPOD_API_KEY",
    "WHISPER_RUNPOD_ID",
    "GENIUS_API_TOKEN",
    "SPOTIFY_COOKIE_SP_DC",
    "RAPIDAPI_KEY",
]

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>karaoke-gen Local WebUI</title>
  <style>
    :root {
      --bg: #0c1017;
      --panel: #161d2a;
      --panel-2: #1d2637;
      --text: #edf2f7;
      --muted: #9fb0c8;
      --accent: #6ee7b7;
      --accent-2: #1f9d78;
      --border: #2a364b;
      --error: #fca5a5;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(110, 231, 183, 0.12), transparent 28%),
        linear-gradient(180deg, #0c1017, #101827 40%, #0c1017 100%);
      color: var(--text);
    }
    main {
      max-width: 1000px;
      margin: 0 auto;
      padding: 32px 18px 64px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: clamp(28px, 5vw, 44px);
      letter-spacing: 0.02em;
    }
    p.lead {
      margin: 0 0 28px;
      color: var(--muted);
      max-width: 760px;
      line-height: 1.6;
    }
    .layout {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 20px;
    }
    .card {
      background: rgba(22, 29, 42, 0.92);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
    }
    .card h2 {
      margin: 0 0 14px;
      font-size: 18px;
      letter-spacing: 0.01em;
    }
    label {
      display: block;
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    input, select, button, textarea {
      font: inherit;
    }
    input[type="text"], input[type="file"], select {
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      padding: 12px 14px;
      margin: 0 0 14px;
    }
    input[type="file"] {
      padding: 10px 12px;
    }
    .hint {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      margin: -6px 0 16px;
    }
    button {
      width: 100%;
      border: 0;
      border-radius: 999px;
      padding: 14px 18px;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: #062a20;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.55;
      cursor: wait;
    }
    .status-row {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      margin-bottom: 14px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--border);
      color: var(--muted);
      background: rgba(255, 255, 255, 0.02);
      font-size: 13px;
    }
    .pill strong {
      color: var(--text);
    }
    .error {
      color: var(--error);
      margin-top: 10px;
      white-space: pre-wrap;
    }
    #outputs a {
      display: block;
      color: var(--accent);
      text-decoration: none;
      padding: 8px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      word-break: break-all;
    }
    #outputs a:last-child {
      border-bottom: 0;
    }
    #log {
      width: 100%;
      min-height: 300px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #0a0f18;
      color: #d7e1f0;
      padding: 14px;
      resize: vertical;
      line-height: 1.4;
    }
    video {
      width: 100%;
      max-height: 420px;
      margin-top: 16px;
      border-radius: 16px;
      background: #000;
    }
    @media (max-width: 860px) {
      .layout {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main>
    <h1>Offline Karaoke Generator</h1>
    <p class="lead">
      Upload local audio and an optional lyrics file, then run the full karaoke generation pipeline
      without AudioShake, RunPod, or online lyrics APIs. This path uses Local Whisper for word-level timing
      and delivers a 720p MP4 that preserves the original vocals by default, without title or end-card overlays.
    </p>

    <div class="layout">
      <section class="card">
        <h2>Create Job</h2>
        <form id="job-form">
          <label for="audio_file">Audio File</label>
          <input id="audio_file" name="audio_file" type="file" accept=".mp3,.wav,.flac,.m4a,.ogg,.aac,audio/*" required />

          <label for="lyrics_file">Lyrics File</label>
          <input id="lyrics_file" name="lyrics_file" type="file" accept=".txt,.lrc,.docx,.rtf" />
          <div class="hint">Recommended for best word alignment, furigana placement, karaoke styling, and highlight accuracy.</div>

          <label for="artist">Artist</label>
          <input id="artist" name="artist" type="text" placeholder="Optional, derived from filename if omitted" />

          <label for="title">Title</label>
          <input id="title" name="title" type="text" placeholder="Optional, derived from filename if omitted" />

          <label for="whisper_model_size">Whisper Model</label>
          <select id="whisper_model_size" name="whisper_model_size">
            <option value="medium">medium</option>
            <option value="small">small</option>
            <option value="base">base</option>
            <option value="tiny">tiny</option>
            <option value="large">large</option>
          </select>

          <label for="whisper_device">Device</label>
          <select id="whisper_device" name="whisper_device">
            <option value="">auto</option>
            <option value="cpu">cpu</option>
            <option value="cuda">cuda</option>
            <option value="mps">mps</option>
          </select>

          <button id="submit-button" type="submit">Generate Offline Karaoke</button>
          <div id="form-error" class="error"></div>
        </form>
      </section>

      <section class="card">
        <h2>Job Status</h2>
        <div class="status-row">
          <div class="pill"><strong>ID</strong> <span id="job-id">-</span></div>
          <div class="pill"><strong>Status</strong> <span id="job-status">idle</span></div>
          <div class="pill"><strong>Track</strong> <span id="job-track">-</span></div>
        </div>

        <div id="job-error" class="error"></div>
        <div id="outputs"></div>
        <video id="preview-video" controls hidden></video>

        <label for="log">Live Log</label>
        <textarea id="log" readonly></textarea>
      </section>
    </div>
  </main>

  <script>
    const form = document.getElementById('job-form');
    const submitButton = document.getElementById('submit-button');
    const formError = document.getElementById('form-error');
    const jobIdEl = document.getElementById('job-id');
    const jobStatusEl = document.getElementById('job-status');
    const jobTrackEl = document.getElementById('job-track');
    const jobErrorEl = document.getElementById('job-error');
    const outputsEl = document.getElementById('outputs');
    const logEl = document.getElementById('log');
    const previewVideo = document.getElementById('preview-video');
    let currentJobId = null;
    let pollTimer = null;

    function resetView() {
      jobErrorEl.textContent = '';
      outputsEl.innerHTML = '';
      logEl.value = '';
      previewVideo.hidden = true;
      previewVideo.removeAttribute('src');
    }

    function renderOutputs(job) {
      outputsEl.innerHTML = '';
      if (!job.outputs || job.outputs.length === 0) {
        return;
      }

      job.outputs.forEach((output) => {
        const link = document.createElement('a');
        link.href = output.url;
        link.textContent = output.label;
        link.target = '_blank';
        outputsEl.appendChild(link);
      });

      if (job.primary_video_url) {
        previewVideo.src = job.primary_video_url;
        previewVideo.hidden = false;
      }
    }

    async function refreshJob(jobId) {
      const response = await fetch(`/api/jobs/${jobId}`);
      if (!response.ok) {
        throw new Error(`Failed to load job ${jobId}`);
      }

      const job = await response.json();
      jobIdEl.textContent = job.job_id;
      jobStatusEl.textContent = job.status;
      jobTrackEl.textContent = `${job.artist} - ${job.title}`;
      jobErrorEl.textContent = job.error || '';
      logEl.value = job.log_tail || '';
      renderOutputs(job);

      if (job.status === 'running') {
        pollTimer = window.setTimeout(() => refreshJob(jobId), 2000);
      } else {
        submitButton.disabled = false;
      }
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      formError.textContent = '';
      jobErrorEl.textContent = '';
      resetView();
      submitButton.disabled = true;

      if (pollTimer) {
        window.clearTimeout(pollTimer);
      }

      try {
        const response = await fetch('/api/jobs', {
          method: 'POST',
          body: new FormData(form),
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || payload.error || 'Failed to create job');
        }
        currentJobId = payload.job_id;
        await refreshJob(currentJobId);
      } catch (error) {
        submitButton.disabled = false;
        formError.textContent = error.message;
      }
    });
  </script>
</body>
</html>
"""


@dataclass
class LocalJob:
    job_id: str
    artist: str
    title: str
    created_at: str
    work_dir: Path
    upload_dir: Path
    output_dir: Path
    log_path: Path
    command: List[str]
    env_overrides: Dict[str, str]
    process: Optional[subprocess.Popen] = None
    status: str = "running"
    returncode: Optional[int] = None
    error: Optional[str] = None
    outputs: List[Dict[str, str]] = field(default_factory=list)
    primary_video_url: Optional[str] = None


class LocalWebUIServer:
    def __init__(self, base_dir: Optional[Path] = None):
        self._ensure_fastapi_available()
        self.base_dir = Path(
            base_dir
            or os.environ.get("KARAOKE_GEN_WEBUI_DIR")
            or os.path.join(tempfile.gettempdir(), "karaoke-gen-webui")
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, LocalJob] = {}
        self.app = FastAPI(title="karaoke-gen Local WebUI")
        self._register_routes()

    def _ensure_fastapi_available(self) -> None:
        if _FASTAPI_IMPORT_ERROR is not None:
            raise RuntimeError(
                "karaoke-gen-webui requires FastAPI and uvicorn. "
                "Install project dependencies before starting the WebUI."
            ) from _FASTAPI_IMPORT_ERROR

    def _register_routes(self) -> None:
        @self.app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            return HTMLResponse(INDEX_HTML)

        @self.app.get("/api/health")
        async def health() -> Dict[str, str]:
            return {"status": "ok"}

        @self.app.post("/api/jobs")
        async def create_job(
            audio_file: UploadFile = File(...),
            lyrics_file: Optional[UploadFile] = File(None),
            artist: str = Form(""),
            title: str = Form(""),
            whisper_model_size: str = Form("medium"),
            whisper_device: str = Form(""),
        ) -> JSONResponse:
            job = await self._create_job(
                audio_file=audio_file,
                lyrics_file=lyrics_file,
                artist=artist,
                title=title,
                whisper_model_size=whisper_model_size,
                whisper_device=whisper_device,
            )
            return JSONResponse(self._serialize_job(job))

        @self.app.get("/api/jobs/{job_id}")
        async def get_job(job_id: str) -> JSONResponse:
            job = self.jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            self._refresh_job(job)
            return JSONResponse(self._serialize_job(job))

        @self.app.get("/api/jobs/{job_id}/files/{relative_path:path}")
        async def get_output_file(job_id: str, relative_path: str):
            job = self.jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")

            target_path = self._resolve_job_path(job.output_dir, relative_path)
            if not target_path.is_file():
                raise HTTPException(status_code=404, detail="File not found")

            return FileResponse(target_path)

    async def _create_job(
        self,
        audio_file: UploadFile,
        lyrics_file: Optional[UploadFile],
        artist: str,
        title: str,
        whisper_model_size: str,
        whisper_device: str,
    ) -> LocalJob:
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="Audio file is required")

        job_id = uuid.uuid4().hex[:8]
        work_dir = self.base_dir / job_id
        upload_dir = work_dir / "uploads"
        output_dir = work_dir / "output"
        log_path = work_dir / "job.log"

        upload_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_path = await self._save_upload(upload_dir, audio_file, "audio")
        lyrics_path = await self._save_upload(upload_dir, lyrics_file, "lyrics") if lyrics_file and lyrics_file.filename else None

        track_artist, track_title = self._derive_metadata(audio_path.stem, artist, title)
        command = self._build_command(
            audio_path=audio_path,
            output_dir=output_dir,
            artist=track_artist,
            title=track_title,
            lyrics_path=lyrics_path,
        )
        env = self._build_environment(whisper_model_size=whisper_model_size, whisper_device=whisper_device)

        log_handle = open(log_path, "w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=os.getcwd(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        log_handle.close()

        job = LocalJob(
            job_id=job_id,
            artist=track_artist,
            title=track_title,
            created_at=datetime.now(timezone.utc).isoformat(),
            work_dir=work_dir,
            upload_dir=upload_dir,
            output_dir=output_dir,
            log_path=log_path,
            command=command,
            env_overrides={
                "WHISPER_MODEL_SIZE": whisper_model_size,
                "WHISPER_DEVICE": whisper_device or "auto",
            },
            process=process,
        )
        self.jobs[job_id] = job
        self._refresh_job(job)
        return job

    async def _save_upload(self, target_dir: Path, upload: UploadFile, prefix: str) -> Path:
        filename = sanitize_filename(upload.filename or f"{prefix}.bin") or f"{prefix}.bin"
        path = target_dir / filename
        with open(path, "wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        await upload.close()
        return path

    def _derive_metadata(self, stem: str, artist: str, title: str) -> tuple[str, str]:
        stem = sanitize_filename(stem) or "Local Track"
        artist = sanitize_filename(artist.strip()) if artist else ""
        title = sanitize_filename(title.strip()) if title else ""

        if artist and title:
            return artist, title

        if " - " in stem:
            guessed_artist, guessed_title = [part.strip() for part in stem.split(" - ", 1)]
        else:
            guessed_artist, guessed_title = "Local Artist", stem

        return artist or guessed_artist, title or guessed_title

    def _build_command(
        self,
        audio_path: Path,
        output_dir: Path,
        artist: str,
        title: str,
        lyrics_path: Optional[Path],
    ) -> List[str]:
        command = [
            sys.executable,
            "-m",
            "karaoke_gen.utils.gen_cli",
            str(audio_path),
            artist,
            title,
            "--output_dir",
            str(output_dir),
            "--skip_transcription_review",
            "--skip_instrumental_review",
            "--yes",
            "--offline",
            "--log_level",
            "info",
        ]
        if lyrics_path:
            command.extend(["--lyrics_file", str(lyrics_path)])
        return command

    def _build_environment(self, whisper_model_size: str, whisper_device: str) -> Dict[str, str]:
        env = os.environ.copy()
        for key in OFFLINE_DISABLED_ENV_VARS:
            env.pop(key, None)

        env["USE_AGENTIC_AI"] = "0"
        env["SKIP_CORRECTION"] = "false"
        env["PYTHONUNBUFFERED"] = "1"
        env["WHISPER_MODEL_SIZE"] = whisper_model_size or "medium"

        if whisper_device:
            env["WHISPER_DEVICE"] = whisper_device
        else:
            env.pop("WHISPER_DEVICE", None)

        return env

    def _refresh_job(self, job: LocalJob) -> None:
        if job.process and job.status == "running":
            returncode = job.process.poll()
            if returncode is None:
                return

            job.returncode = returncode
            if returncode == 0:
                job.outputs = self._collect_outputs(job)
                primary_video = self._find_primary_video(job)
                if primary_video is None:
                    job.status = "failed"
                    job.error = "karaoke-gen finished but did not produce a playable default with-vocals MP4 output"
                else:
                    job.status = "completed"
                    job.error = None
                    job.primary_video_url = primary_video["url"]
            else:
                job.status = "failed"
                job.error = f"karaoke-gen exited with status {returncode}"

    def _collect_outputs(self, job: LocalJob) -> List[Dict[str, str]]:
        files: List[Dict[str, str]] = []
        kind_by_suffix = {
            ".mp4": "video",
            ".mkv": "video",
            ".lrc": "lyrics",
            ".ass": "lyrics",
            ".txt": "text",
            ".json": "json",
            ".zip": "archive",
        }
        candidates = sorted(
            path for path in job.output_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in kind_by_suffix
        )

        for path in candidates:
            relative_path = str(path.relative_to(job.output_dir))
            files.append(
                {
                    "kind": kind_by_suffix[path.suffix.lower()],
                    "label": relative_path,
                    "url": f"/api/jobs/{job.job_id}/files/{relative_path}",
                }
            )

        files.sort(key=self._output_sort_key)
        return files

    def _output_sort_key(self, item: Dict[str, str]) -> tuple[int, str]:
        label = item["label"].lower()
        if "final karaoke lossy 720p" in label or "(lossy 720p).mp4" in label:
            return (0, label)
        if "(with vocals).mp4" in label or "with_vocals.mp4" in label:
            return (1, label)
        if "(with vocals).mkv" in label or "(with vocals).mov" in label or "with_vocals." in label:
            return (2, label)
        if "(karaoke).mp4" in label:
            return (3, label)
        if "(lossless 4k).mp4" in label:
            return (4, label)
        if "(lossy 4k).mp4" in label:
            return (5, label)
        if item["kind"] == "video":
            return (6, label)
        return (7, label)

    def _is_default_vocals_video(self, item: Dict[str, str]) -> bool:
        label = item["label"].lower()
        return item["kind"] == "video" and label.endswith(".mp4") and (
            "final karaoke lossy 720p" in label
            or "(lossy 720p).mp4" in label
            or "(with vocals).mp4" in label
            or "with_vocals.mp4" in label
        )

    def _find_primary_video(self, job: LocalJob) -> Optional[Dict[str, str]]:
        default_outputs = [item for item in job.outputs if self._is_default_vocals_video(item)]

        for item in default_outputs:
            path = self._resolve_job_path(job.output_dir, item["label"])
            if self._is_playable_video(path):
                return item
        return None

    def _is_playable_video(self, path: Path) -> bool:
        if not path.is_file() or path.stat().st_size <= 0:
            return False

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-print_format",
                    "json",
                    "-show_entries",
                    "format=duration:stream=codec_type",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

        try:
            probe = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return False

        streams = probe.get("streams", [])
        has_video_stream = any(stream.get("codec_type") == "video" for stream in streams)

        duration_raw = probe.get("format", {}).get("duration")
        try:
            duration = float(duration_raw)
        except (TypeError, ValueError):
            duration = 0.0

        return has_video_stream and duration > 0

    def _serialize_job(self, job: LocalJob) -> Dict[str, object]:
        return {
            "job_id": job.job_id,
            "artist": job.artist,
            "title": job.title,
            "status": job.status,
            "created_at": job.created_at,
            "error": job.error,
            "returncode": job.returncode,
            "command": job.command,
            "outputs": job.outputs,
            "primary_video_url": job.primary_video_url,
            "log_tail": self._read_log_tail(job.log_path),
        }

    def _read_log_tail(self, path: Path, max_bytes: int = 16000) -> str:
        if not path.exists():
            return ""

        size = path.stat().st_size
        with open(path, "rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
            content = handle.read().decode("utf-8", errors="replace")
        return content

    def _resolve_job_path(self, root: Path, relative_path: str) -> Path:
        resolved = (root / relative_path).resolve()
        root_resolved = root.resolve()
        if resolved != root_resolved and root_resolved not in resolved.parents:
            raise HTTPException(status_code=400, detail="Invalid file path")
        return resolved


def create_app(base_dir: Optional[Path] = None) -> FastAPI:
    return LocalWebUIServer(base_dir=base_dir).app


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency availability is environment-specific
        raise RuntimeError(
            "karaoke-gen-webui requires uvicorn. Install project dependencies before starting the WebUI."
        ) from exc

    host = os.environ.get("KARAOKE_GEN_WEBUI_HOST", "127.0.0.1")
    port = int(os.environ.get("KARAOKE_GEN_WEBUI_PORT", "8780"))
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
