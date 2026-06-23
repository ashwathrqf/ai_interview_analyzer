"""Feature 10: Resume-specific interview evaluation.

Compares the candidate's spoken transcript against their resume to flag
claims on the resume that weren't substantiated when actually asked about
in the interview (e.g. resume says "built an event-driven architecture"
but the candidate never explained that when prompted).
"""

import json
import re
from difflib import SequenceMatcher

import pdfplumber

from llm.client import chat


SYSTEM_PROMPT = """You are an experienced interview coach. You are given a \
candidate's resume and their transcribed spoken answer to an interview \
question. Your job is to check whether claims, projects, or skills listed \
on the resume that are RELEVANT to the question were actually explained or \
substantiated in the spoken answer.

The transcript was produced by automatic speech-to-text and may contain \
minor errors or garbled words. Do not penalize transcription artifacts; \
focus on substance.

Be strict about what counts as "well substantiated": only include a resume \
item there if the candidate explicitly named or clearly described that \
specific project, skill, or achievement in their spoken answer. A vague, \
generic statement (e.g. "I'm interested in machine learning" or "I'm \
honest and hardworking") does NOT substantiate a specific resume claim \
(e.g. a named project, a specific technique, or a quantified result) even \
if they are thematically related. When in doubt, treat the claim as \
unsubstantiated rather than well substantiated.

Only flag a resume claim as a gap if it is clearly relevant to the question \
asked and the candidate had a natural opportunity to mention it but didn't, \
or mentioned it without explaining it. Do not flag claims that are simply \
unrelated to the question.

Respond with ONLY a JSON object, no other text, no markdown formatting, no \
backticks. The JSON object must have exactly these keys:
{
  "unsubstantiated_claims": ["short description of a resume claim not explained", ...],
  "well_substantiated": ["short description of a resume claim that WAS explained well", ...],
  "summary": "1-2 sentence overall summary of how well the answer drew on the resume"
}

If there are no relevant gaps, return an empty list for unsubstantiated_claims. \
If nothing was clearly and specifically substantiated, return an empty list \
for well_substantiated too — do not force matches that aren't really there. \
Be specific — reference the actual project/skill name from the resume, not generic language."""


USER_PROMPT_TEMPLATE = """Interview question: {question}

Candidate's resume:
{resume_text}

Candidate's transcribed spoken answer:
{transcript}

Evaluate how well the spoken answer substantiates relevant resume claims, \
and respond with the JSON object as instructed."""


def extract_resume_text(pdf_path: str) -> str:
    """Extract plain text from a resume PDF.

    Uses pdfplumber for better layout handling than basic PDF text
    extraction, which matters for resumes with columns or tight spacing.
    """
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def _extract_json(raw_response: str) -> dict:
    match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model response: {raw_response!r}")
    return json.loads(match.group(0))


def _significant_words(text: str) -> set:
    """Return lowercased words from text, excluding short/common filler words,
    used for a lightweight overlap check between a claim and the transcript.
    """
    stopwords = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
        "is", "was", "were", "be", "been", "this", "that", "i", "my", "me",
        "it", "as", "at", "by", "from", "using",
    }
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in stopwords and len(w) > 2}


def _has_transcript_support(claim: str, transcript: str, min_overlap: int = 2) -> bool:
    """Check whether a claim has at least minimal real support in the transcript.

    This guards against the LLM crediting a resume claim as 'well
    substantiated' just because it appears prominently in the resume,
    without the candidate actually having said anything resembling it.
    Uses significant-word overlap rather than exact substring matching,
    since the model may paraphrase slightly even when correctly grounded.
    """
    claim_words = _significant_words(claim)
    transcript_words = _significant_words(transcript)
    overlap = claim_words & transcript_words
    return len(overlap) >= min_overlap


def _verify_well_substantiated(result: dict, transcript: str, resume_text: str) -> dict:
    """Filter out 'well_substantiated' claims that either:
    (a) don't have real word overlap with the transcript (LLM credited a
        resume claim without the candidate actually saying anything like it), or
    (b) don't have real word overlap with the resume (LLM echoed a transcript
        phrase back as if it were a resume claim, which isn't a real match).
    Both checks must pass for a claim to stay in well_substantiated.
    """
    verified = []
    demoted = []

    for claim in result.get("well_substantiated", []):
        supported_by_transcript = _has_transcript_support(claim, transcript)
        supported_by_resume = _has_transcript_support(claim, resume_text, min_overlap=2)

        if supported_by_transcript and supported_by_resume:
            verified.append(claim)
        else:
            demoted.append(claim)

    result["well_substantiated"] = verified
    if demoted:
        result["unsubstantiated_claims"] = result.get("unsubstantiated_claims", []) + demoted

    return result


def evaluate_against_resume(resume_text: str, transcript: str, question: str) -> dict:
    """Compare transcript against resume content for a given question.

    Returns a dict like:
    {
        "unsubstantiated_claims": [...],
        "well_substantiated": [...],
        "summary": str
    }
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question, resume_text=resume_text, transcript=transcript
    )
    raw_response = chat(SYSTEM_PROMPT, user_prompt)

    try:
        result = _extract_json(raw_response)
    except (ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"Failed to parse LLM response as JSON. Raw response was:\n{raw_response}"
        ) from e

    required_keys = {"unsubstantiated_claims", "well_substantiated", "summary"}
    missing = required_keys - result.keys()
    if missing:
        raise RuntimeError(f"LLM response missing required keys: {missing}")

    result = _verify_well_substantiated(result, transcript, resume_text)

    return result


if __name__ == "__main__":
    # Update this path to point at your resume PDF, placed in data/sample_videos/
    # (or wherever you'd like to keep test files — that folder is gitignored).
    resume_path = "data/sample_videos/resume.pdf"

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

    print("Extracting resume text...")
    resume_text = extract_resume_text(resume_path)
    print(f"Extracted {len(resume_text)} characters from resume.\n")
    print("--- RESUME TEXT PREVIEW (first 500 chars) ---")
    print(resume_text[:500])

    print("\nSending to local LLM, this may take a moment...\n")
    result = evaluate_against_resume(resume_text, sample_transcript, sample_question)

    print("--- UNSUBSTANTIATED CLAIMS ---")
    for c in result["unsubstantiated_claims"]:
        print(f"- {c}")

    print("\n--- WELL SUBSTANTIATED ---")
    for c in result["well_substantiated"]:
        print(f"- {c}")

    print(f"\n--- SUMMARY ---\n{result['summary']}")