"""Detect and count filler words in a transcript."""

import re

FILLER_WORDS = ["uh", "umm", "um", "like", "you know", "basically", "actually"]


def count_filler_words(transcript: str) -> dict:
    """Count occurrences of each filler word in transcript.

    Matching is case-insensitive and uses word boundaries so "like" doesn't
    match inside "unlike", etc. Multi-word fillers ("you know") are matched
    as phrases.

    Returns a dict like:
    {"uh": 3, "umm": 0, "um": 2, "like": 5, "you know": 1,
     "basically": 0, "actually": 1, "total": 12}
    """
    text_lower = transcript.lower()
    counts = {}

    for filler in FILLER_WORDS:
        # \b word boundaries so we don't match substrings inside other words
        pattern = r"\b" + re.escape(filler) + r"\b"
        matches = re.findall(pattern, text_lower)
        counts[filler] = len(matches)

    counts["total"] = sum(counts.values())
    return counts


if __name__ == "__main__":
    # Quick manual test using the real transcript from the previous step.
    sample_transcript = (
        "Hi, I'm Ashwath. I'm studying Mechanical Engineering in IIT Madras. "
        "I live in Bangalore, my favorite hobbies are reading books, going "
        "through YouTube stuff, any new challenges and creating projects "
        "like this. I love doing stuff like this. I want to get into "
        "quantitative finance in NFO or machine learning and data analytics. "
        "And my priority of the night I may say is my honesty on my talent "
        "to lead a team and to work under pressure. Thank you."
    )
    result = count_filler_words(sample_transcript)
    print(result)