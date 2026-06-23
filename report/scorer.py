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
