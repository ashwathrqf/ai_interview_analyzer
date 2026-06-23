# Run from the root of your cloned repo: .\scaffold.ps1
# If blocked by execution policy, run: powershell -ExecutionPolicy Bypass -File .\scaffold.ps1

$folders = @(
    "audio", "speech", "vision", "llm", "report", "utils",
    "data\sample_videos", "tests"
)
foreach ($f in $folders) {
    New-Item -ItemType Directory -Force -Path $f | Out-Null
}

# __init__.py for each package
$initPkgs = @("audio", "speech", "vision", "llm", "report", "utils")
foreach ($p in $initPkgs) {
    New-Item -ItemType File -Force -Path "$p\__init__.py" | Out-Null
}

# audio/extract_audio.py
@'
"""Extract audio track from an uploaded interview video."""


def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio from video_path and save as a wav/mp3 at output_path.

    Returns the output_path on success.
    """
    raise NotImplementedError
'@ | Set-Content "audio\extract_audio.py"

# speech/transcribe.py
@'
"""Speech-to-text using Whisper."""


def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio_path using Whisper.

    Returns a dict with at least: {"text": str, "segments": list}
    """
    raise NotImplementedError
'@ | Set-Content "speech\transcribe.py"

# speech/filler_words.py
@'
"""Detect and count filler words in a transcript."""

FILLER_WORDS = ["uh", "umm", "um", "like", "you know", "basically", "actually"]


def count_filler_words(transcript: str) -> dict:
    """Count occurrences of each filler word in transcript.

    Returns a dict like {"uh": 3, "umm": 5, "total": 17}
    """
    raise NotImplementedError
'@ | Set-Content "speech\filler_words.py"

# speech/speech_metrics.py
@'
"""Speaking speed (WPM) and pause detection metrics."""


def calculate_wpm(transcript: str, duration_seconds: float) -> float:
    """Return words-per-minute for the given transcript and duration."""
    raise NotImplementedError


def detect_pauses(segments: list) -> dict:
    """Detect pauses between Whisper segments.

    Returns a dict like {"average_pause": 0.74, "pause_count": 12}
    """
    raise NotImplementedError
'@ | Set-Content "speech\speech_metrics.py"

# vision/eye_contact.py
@'
"""Eye contact estimation using MediaPipe FaceMesh."""


def estimate_eye_contact(video_path: str) -> float:
    """Return percentage of frames where the subject is looking at the camera."""
    raise NotImplementedError
'@ | Set-Content "vision\eye_contact.py"

# vision/smile_detection.py
@'
"""Smile frequency detection."""


def detect_smile_frequency(video_path: str) -> float:
    """Return percentage of frames where the subject is smiling."""
    raise NotImplementedError
'@ | Set-Content "vision\smile_detection.py"

# vision/head_pose.py
@'
"""Head pose estimation (looking down/left/right)."""


def estimate_head_pose(video_path: str) -> dict:
    """Return breakdown of head pose direction percentages over the video."""
    raise NotImplementedError
'@ | Set-Content "vision\head_pose.py"

# llm/client.py
@'
"""OpenAI client wrapper. Reads API key from environment via dotenv."""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Copy .env.example to .env and fill it in.")
        _client = OpenAI(api_key=api_key)
    return _client
'@ | Set-Content "llm\client.py"

# llm/evaluate_answer.py
@'
"""Feature 9: LLM evaluation of the interview answer."""


def evaluate_answer(question: str, transcript: str) -> dict:
    """Send question + transcript to the LLM and return structured feedback.

    Returns a dict like:
    {
        "strengths": [...],
        "weaknesses": [...],
        "improved_answer": str,
        "rating": int
    }
    """
    raise NotImplementedError
'@ | Set-Content "llm\evaluate_answer.py"

# llm/resume_match.py
@'
"""Feature 10: Resume-specific interview evaluation."""


def evaluate_against_resume(resume_text: str, transcript: str) -> dict:
    """Compare transcript against resume content.

    Returns a dict highlighting resume claims not adequately explained in the interview.
    """
    raise NotImplementedError
'@ | Set-Content "llm\resume_match.py"

# report/scorer.py
@'
"""Combine all metrics into category and overall scores."""


def calculate_overall_score(speech_metrics: dict, vision_metrics: dict, llm_feedback: dict) -> dict:
    """Combine metrics into a final score breakdown.

    Returns a dict like:
    {
        "overall": 87,
        "confidence": 90,
        "communication": 85,
        "body_language": 88,
        "technical": 84
    }
    """
    raise NotImplementedError
'@ | Set-Content "report\scorer.py"

# report/pdf_export.py
@'
"""Export the final report as a downloadable PDF."""


def export_report_pdf(report_data: dict, output_path: str) -> str:
    """Generate a PDF report from report_data and save to output_path."""
    raise NotImplementedError
'@ | Set-Content "report\pdf_export.py"

# utils/config.py
@'
"""Central config and constants."""

IDEAL_WPM_MIN = 120
IDEAL_WPM_MAX = 160

FILLER_WORDS = ["uh", "umm", "um", "like", "you know", "basically", "actually"]
'@ | Set-Content "utils\config.py"

# tests/test_speech_metrics.py
@'
"""Basic tests for speech metrics."""

import pytest

from speech.speech_metrics import calculate_wpm


def test_calculate_wpm_placeholder():
    with pytest.raises(NotImplementedError):
        calculate_wpm("hello world this is a test", 10)
'@ | Set-Content "tests\test_speech_metrics.py"

# root files
New-Item -ItemType File -Force -Path "streamlit_app.py" | Out-Null

if (-not (Test-Path ".env.example")) {
@'
OPENAI_API_KEY=
'@ | Set-Content ".env.example"
}

if (-not (Test-Path "README.md")) {
@'
# InterviewIQ AI

AI-powered multimodal mock interview analyzer. Upload an interview video and get
feedback on speech, body language, and answer quality, scored and combined into
a final report.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env  # then fill in OPENAI_API_KEY
```

## Run

```bash
streamlit run streamlit_app.py
```
'@ | Set-Content "README.md"
}

Write-Host "Done. Folder structure created."