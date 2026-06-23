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
