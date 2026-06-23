"""Feature 9: LLM evaluation of the interview answer, using a local Ollama
model (no API key, no cost). Returns structured feedback: strengths,
weaknesses, an improved version of the answer, and a numeric rating.
"""

import json
import re

from llm.client import chat


SYSTEM_PROMPT = """You are an experienced interview coach giving feedback on \
a candidate's spoken answer to an interview question.

The answer was transcribed from speech using automatic transcription, so it \
may contain small errors, garbled words, or odd phrasing that are \
transcription artifacts, NOT mistakes the candidate actually made. Do not \
penalize or comment on minor transcription glitches (e.g. a slightly wrong \
word that obviously should be something else in context). Focus on the \
substance, structure, and content of the answer.

Respond with ONLY a JSON object, no other text, no markdown formatting, no \
backticks. The JSON object must have exactly these keys:
{
  "strengths": ["short point", "short point", ...],
  "weaknesses": ["short point", "short point", ...],
  "improved_answer": "a rewritten, improved version of the answer, 2-4 sentences",
  "rating": <integer from 1 to 10>
}

Give 2-4 strengths and 2-4 weaknesses, each as a short, specific sentence. \
Be honest and constructive, not just generically positive."""


USER_PROMPT_TEMPLATE = """Interview question: {question}

Candidate's transcribed answer: {transcript}

Evaluate this answer and respond with the JSON object as instructed."""


def _extract_json(raw_response: str) -> dict:
    """Extract a JSON object from the model's raw text response.

    Models sometimes wrap JSON in markdown code fences or add stray text
    even when instructed not to, so we extract the first {...} block
    rather than assuming the response is pure JSON.
    """
    match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model response: {raw_response!r}")
    return json.loads(match.group(0))


def evaluate_answer(question: str, transcript: str) -> dict:
    """Send question + transcript to the local LLM and return structured feedback.

    Returns a dict like:
    {
        "strengths": [...],
        "weaknesses": [...],
        "improved_answer": str,
        "rating": int
    }
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(question=question, transcript=transcript)
    raw_response = chat(SYSTEM_PROMPT, user_prompt)

    try:
        result = _extract_json(raw_response)
    except (ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"Failed to parse LLM response as JSON. Raw response was:\n{raw_response}"
        ) from e

    # Basic validation so callers can trust the shape of the result
    required_keys = {"strengths", "weaknesses", "improved_answer", "rating"}
    missing = required_keys - result.keys()
    if missing:
        raise RuntimeError(f"LLM response missing required keys: {missing}")

    return result


if __name__ == "__main__":
    # Quick manual test using the real transcript from earlier steps.
    sample_question = "Tell me about yourself."
    sample_transcript = (
        "Hi, I'm Ashwath. I'm studying Mechanical Engineering in IIT Madras. "
        "I live in Bangalore, my favorite hobbies are reading books, going "
        "through YouTube stuff, any new challenges and creating projects "
        "like this. I love doing stuff like this. I want to get into "
        "quantitative finance in NFO or machine learning and data analytics. "
        "And my priority of the night I may say is my honesty on my talent "
        "to lead a team and to work under pressure. Thank you."
    )

    print("Sending to local LLM, this may take a moment...\n")
    result = evaluate_answer(sample_question, sample_transcript)

    print("--- STRENGTHS ---")
    for s in result["strengths"]:
        print(f"- {s}")

    print("\n--- WEAKNESSES ---")
    for w in result["weaknesses"]:
        print(f"- {w}")

    print(f"\n--- IMPROVED ANSWER ---\n{result['improved_answer']}")
    print(f"\n--- RATING ---\n{result['rating']}/10")