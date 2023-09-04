import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from download import download_model_if_not_cached

app = FastAPI()

# url http://127.0.0.1:8000/v1/audio/transcriptions \
#  -H "Authorization: Bearer $OPENAI_API_KEY" \
#  -H "Content-Type: multipart/form-data" \
#  -F model="whisper-ch" \
#  -F file="@/path/to/file/openai.mp3"

# {
#  "text": "Imagine the wildest idea that you've ever had, and you're curious about how it might scale to something that's a 100, a 1,000 times bigger..."
# }

MODEL_DATA_DIR = "/data/cache"


@lru_cache(maxsize=1)
def get_whisper_model(whisper_model: str, quantization: str) -> WhisperModel:
    """Get a whisper model from the cache or download it if it doesn't exist"""

    model_folder = download_model_if_not_cached(
        model_data_dir=MODEL_DATA_DIR,
        whisper_model_name=whisper_model,
        quantization=quantization,
    )

    model = WhisperModel(str(model_folder), compute_type=quantization)
    return model


def transcribe(
    audio_path: str, whisper_model: str, quantization: str, **whisper_args
) -> Iterable[Segment]:
    """Transcribe the audio file using whisper"""

    # Get whisper model
    # NOTE: If mulitple models are selected, this may keep all of them in memory depending on the cache size
    transcriber = get_whisper_model(whisper_model, quantization)

    segments, _ = transcriber.transcribe(
        audio=audio_path,
        **whisper_args,
    )

    return segments


WHISPER_DEFAULT_SETTINGS = {
    "whisper_model": os.getenv("MODEL"),
    "quantization": os.getenv("QUANTIZATION"),
    "task": "transcribe",
    "language": "de",
    "beam_size": 5,
}
UPLOAD_DIR = "/data/tmp"


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    model: str = Form(...),
    file: UploadFile = File(...),
    response_format: Optional[str] = Form(None),
):
    assert model == "whisper-ch"
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Bad Request, bad file"
        )
    if response_format is None:
        response_format = "json"
    if response_format not in ["json", "text", "verbose_json"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Bad Request, bad response_format, supported formats are json, verbose_json and text",
        )

    Path(UPLOAD_DIR).mkdir(exist_ok=True, parents=True)
    upload_path = Path(UPLOAD_DIR, file.filename)

    with open(upload_path, "wb+") as upload_file:
        shutil.copyfileobj(file.file, upload_file)

    segments = transcribe(audio_path=str(upload_path), **WHISPER_DEFAULT_SETTINGS)

    os.remove(upload_path)

    segment_dicts = []

    for segment in segments:
        segment_dicts.append(
            {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
                "words": segment.words,
            }
        )

    if response_format in ["text", "json"]:
        text = " ".join([seg["text"].strip() for seg in segment_dicts])

    if response_format in ["verbose_json"]:
        return segment_dicts

    if response_format in ["text"]:
        return text

    return {"text": text}
