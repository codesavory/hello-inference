from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

import modal
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

MAX_BYTES = 10 * 1024 * 1024
VIDEO_DIR = "/vfi-video-store"
MODEL_DIR = "/vfi-models"
ATM_VFI_DIR = "/atm-vfi"
DEFAULT_CHECKPOINT_NAME = "atm-vfi.ckpt"

video_volume = modal.Volume.from_name(
    "styleframe-vfi-storage",
    create_if_missing=True,
)
model_volume = modal.Volume.from_name(
    "styleframe-vfi-models",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "git",
        "libgl1",
        "libglib2.0-0",
        "zlib1g-dev",
        "libjpeg-dev",
    )
    .run_commands(
        "python -m pip install --upgrade pip",
        "git clone --depth 1 https://github.com/Gancheekim/ATM-VFI.git /atm-vfi",
        "pip install --no-cache-dir fastapi>=0.115.0 python-multipart>=0.0.9",
        "pip install --no-cache-dir timm",
        "pip install --no-cache-dir einops",
        "pip install --no-cache-dir -r /atm-vfi/requirements.txt "
        "--extra-index-url https://download.pytorch.org/whl/cu118",
    )
)

app = modal.App("styleframe-atm-vfi", image=image)
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


def _resolve_checkpoint(name: str | None) -> Path:
    checkpoint_name = name or DEFAULT_CHECKPOINT_NAME
    path = Path(MODEL_DIR) / checkpoint_name
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Checkpoint not found. Upload one via POST /vfi/checkpoints or "
                "provide a valid checkpoint name."
            ),
        )
    return path


def _run_atm_vfi(
    input_path: Path,
    output_path: Path,
    model_type: str,
    checkpoint_path: Path,
    combine_video: bool,
) -> None:
    cmd = [
        "python",
        str(Path(ATM_VFI_DIR) / "demo_2x.py"),
        "--model_type",
        model_type,
        "--ckpt",
        str(checkpoint_path),
        "--video",
        str(input_path),
        "--out",
        str(output_path),
    ]
    if combine_video:
        cmd.append("--combine_video")

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ATM_VFI_DIR}{os.pathsep}{existing}" if existing else ATM_VFI_DIR

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=ATM_VFI_DIR,
        env=env,
    )
    if result.returncode != 0:
        detail = result.stderr[-400:] or "ATM-VFI failed"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )


def _run_atm_vfi_frames(
    frame0_path: Path,
    frame1_path: Path,
    output_path: Path,
    model_type: str,
    checkpoint_path: Path,
) -> None:
    cmd = [
        "python",
        str(Path(ATM_VFI_DIR) / "demo_2x.py"),
        "--model_type",
        model_type,
        "--ckpt",
        str(checkpoint_path),
        "--frame0",
        str(frame0_path),
        "--frame1",
        str(frame1_path),
        "--out",
        str(output_path),
    ]

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ATM_VFI_DIR}{os.pathsep}{existing}" if existing else ATM_VFI_DIR

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=ATM_VFI_DIR,
        env=env,
    )
    if result.returncode != 0:
        detail = result.stderr[-400:] or "ATM-VFI failed"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )


@web_app.post("/vfi/checkpoints")
async def upload_checkpoint(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Checkpoint filename is required.",
        )

    dest = Path(MODEL_DIR) / Path(file.filename).name
    with dest.open("wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    model_volume.commit()
    return {"checkpoint": dest.name}


@web_app.post("/vfi/interpolate")
async def interpolate(
    file: UploadFile = File(...),
    model_type: str = Form("base"),
    checkpoint: str | None = Form(None),
    combine_video: str = Form("false"),
):
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only video uploads are supported.",
        )

    normalized_type = model_type.strip().lower()
    if normalized_type not in {"base", "lite"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="model_type must be 'base' or 'lite'.",
        )

    combine_flag = combine_video.strip().lower() in {"1", "true", "yes"}
    checkpoint_path = _resolve_checkpoint(checkpoint)

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

        video_id = uuid.uuid4().hex
        output_path = Path(VIDEO_DIR) / f"{video_id}.mp4"
        _run_atm_vfi(
            input_path,
            output_path,
            normalized_type,
            checkpoint_path,
            combine_flag,
        )
        metadata = _probe_video(output_path)
        video_volume.commit()

    return {
        "id": video_id,
        "metadata": metadata,
    }


@web_app.post("/vfi/interpolate-frames")
async def interpolate_frames(
    frame0: UploadFile = File(...),
    frame1: UploadFile = File(...),
    model_type: str = Form("base"),
    checkpoint: str | None = Form(None),
):
    for frame in (frame0, frame1):
        if frame.content_type and not frame.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only image uploads are supported for frames.",
            )

    normalized_type = model_type.strip().lower()
    if normalized_type not in {"base", "lite"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="model_type must be 'base' or 'lite'.",
        )

    checkpoint_path = _resolve_checkpoint(checkpoint)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        frame0_path = temp_root / (Path(frame0.filename).name or "frame0.png")
        frame1_path = temp_root / (Path(frame1.filename).name or "frame1.png")

        with frame0_path.open("wb") as handle:
            while True:
                chunk = await frame0.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)

        with frame1_path.open("wb") as handle:
            while True:
                chunk = await frame1.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)

        frame_id = uuid.uuid4().hex
        output_path = Path(VIDEO_DIR) / f"{frame_id}.png"
        _run_atm_vfi_frames(
            frame0_path,
            frame1_path,
            output_path,
            normalized_type,
            checkpoint_path,
        )
        video_volume.commit()

    return {"id": frame_id}


@web_app.get("/video/{video_id}")
async def fetch_video(video_id: str):
    path = Path(VIDEO_DIR) / f"{video_id}.mp4"
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return StreamingResponse(path.open("rb"), media_type="video/mp4")


@web_app.get("/frame/{frame_id}")
async def fetch_frame(frame_id: str):
    path = Path(VIDEO_DIR) / f"{frame_id}.png"
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return StreamingResponse(path.open("rb"), media_type="image/png")


@web_app.get("/health")
async def health():
    return {"status": "ok"}


@app.function(
    gpu="A10G",
    volumes={VIDEO_DIR: video_volume, MODEL_DIR: model_volume},
)
@modal.asgi_app()
def fastapi_app():
    return web_app
