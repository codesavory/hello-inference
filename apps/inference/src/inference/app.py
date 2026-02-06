from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

import modal
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

MAX_BYTES = 10 * 1024 * 1024
VIDEO_DIR = "/video-store"

volume = modal.Volume.from_name("styleframe-inference-storage", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi>=0.115.0",
        "python-multipart>=0.0.9",
    )
)

app = modal.App("styleframe-transcoder", image=image)
web_app = FastAPI()

allowed_origins = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_ratio(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        if denominator == "0":
            return 0.0
        return float(numerator) / float(denominator)
    return float(value)


def _probe_video(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ffprobe failed",
        ) from exc

    data = json.loads(result.stdout)
    stream = next(
        (item for item in data.get("streams", []) if item.get("codec_type") == "video"),
        None,
    )

    width = int(stream.get("width", 0)) if stream else 0
    height = int(stream.get("height", 0)) if stream else 0
    frame_rate = _parse_ratio(
        stream.get("avg_frame_rate") if stream else None
    )
    duration_raw = None
    if stream and stream.get("duration"):
        duration_raw = stream.get("duration")
    else:
        duration_raw = data.get("format", {}).get("duration")

    duration = float(duration_raw) if duration_raw else 0.0

    return {
        "width": width,
        "height": height,
        "durationSeconds": duration,
        "frameRate": frame_rate,
    }


def _transcode(input_path: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-vf",
        "format=gray",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcode failed: {result.stderr[-400:]}",
        )


@web_app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only video uploads are supported.",
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "input"
        size = 0
        with input_path.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="File exceeds 10MB limit.",
                    )
                handle.write(chunk)

        metadata = _probe_video(input_path)
        video_id = uuid.uuid4().hex
        output_path = Path(VIDEO_DIR) / f"{video_id}.mp4"
        _transcode(input_path, output_path)
        try:
            input_path.unlink()
        except FileNotFoundError:
            pass
        volume.commit()

    return {
        "id": video_id,
        "metadata": metadata,
    }


@web_app.get("/video/{video_id}")
async def fetch_video(video_id: str):
    path = Path(VIDEO_DIR) / f"{video_id}.mp4"
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return StreamingResponse(path.open("rb"), media_type="video/mp4")


@web_app.get("/health")
async def health():
    return {"status": "ok"}


@app.function(volumes={VIDEO_DIR: volume})
@modal.asgi_app()
def fastapi_app():
    return web_app