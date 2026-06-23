"""Speech-to-text using Whisper."""


def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio_path using Whisper.

    Returns a dict with at least: {"text": str, "segments": list}
    """
    raise NotImplementedError
