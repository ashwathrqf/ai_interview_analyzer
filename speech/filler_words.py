"""Detect and count filler words in a transcript."""

FILLER_WORDS = ["uh", "umm", "um", "like", "you know", "basically", "actually"]


def count_filler_words(transcript: str) -> dict:
    """Count occurrences of each filler word in transcript.

    Returns a dict like {"uh": 3, "umm": 5, "total": 17}
    """
    raise NotImplementedError
