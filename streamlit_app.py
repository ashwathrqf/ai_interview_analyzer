"""InterviewIQ AI — Streamlit dashboard.

Ties together every pipeline module: video upload, audio extraction,
transcription, speech metrics, vision metrics, LLM evaluation, optional
resume-aware evaluation, scoring, and final report display.
"""

import os
import tempfile

import streamlit as st

from audio.extract_audio import extract_audio
from speech.transcribe import transcribe_audio
from speech.filler_words import count_filler_words
from speech.speech_metrics import calculate_wpm, detect_pauses, wpm_feedback
from vision.eye_contact import estimate_eye_contact
from vision.head_pose import estimate_head_pose
from vision.smile_detection import detect_smile_frequency
from llm.evaluate_answer import evaluate_answer
from llm.resume_match import extract_resume_text, evaluate_against_resume
from report.scorer import calculate_overall_score
from report.pdf_export import export_report_pdf


PRESET_QUESTIONS = [
    "Tell me about yourself.",
    "Why do you want to work in quantitative finance?",
    "Describe a challenging project you've worked on.",
    "What are your strengths and weaknesses?",
    "Why should we hire you?",
]


st.set_page_config(page_title="InterviewIQ AI", layout="wide")
st.title("InterviewIQ AI")
st.caption("Multimodal mock interview analyzer — speech, body language, and content feedback")


# --- Input section ---
col1, col2 = st.columns(2)

with col1:
    question = st.selectbox("Question you're answering", PRESET_QUESTIONS)
    video_file = st.file_uploader("Upload your interview video (mp4)", type=["mp4", "mov"])

with col2:
    use_resume = st.checkbox("Also check my answer against my resume (optional)")
    resume_file = None
    if use_resume:
        resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

run_button = st.button("Analyze interview", type="primary", disabled=video_file is None)


# --- Pipeline execution ---
if run_button and video_file is not None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = os.path.join(tmp_dir, "input_video.mp4")
        with open(video_path, "wb") as f:
            f.write(video_file.read())

        audio_path = os.path.join(tmp_dir, "audio.wav")

        progress = st.progress(0, text="Extracting audio...")
        extract_audio(video_path, audio_path)

        progress.progress(15, text="Transcribing speech (this can take a minute)...")
        transcription = transcribe_audio(audio_path)
        transcript_text = transcription["text"].strip()
        segments = transcription["segments"]

        progress.progress(35, text="Analyzing speech patterns...")
        filler_result = count_filler_words(transcript_text)
        word_count = len(transcript_text.split())
        duration = segments[-1]["end"] - segments[0]["start"] if segments else 1.0
        wpm = calculate_wpm(transcript_text, duration)
        pause_result = detect_pauses(segments)

        progress.progress(50, text="Analyzing eye contact and head pose...")
        eye_contact_result = estimate_eye_contact(video_path)
        head_pose_result = estimate_head_pose(video_path)

        progress.progress(65, text="Analyzing facial expressions...")
        smile_result = detect_smile_frequency(video_path)

        progress.progress(75, text="Getting AI feedback on your answer (local LLM, may take a moment)...")
        llm_feedback = evaluate_answer(question, transcript_text)

        resume_result = None
        if use_resume and resume_file is not None:
            progress.progress(90, text="Checking your answer against your resume...")
            resume_path = os.path.join(tmp_dir, "resume.pdf")
            with open(resume_path, "wb") as f:
                f.write(resume_file.read())
            resume_text = extract_resume_text(resume_path)
            resume_result = evaluate_against_resume(resume_text, transcript_text, question)

        progress.progress(95, text="Calculating final scores...")
        speech_metrics_for_scoring = {
            "wpm": wpm,
            "filler_total": filler_result["total"],
            "word_count": word_count,
            "pause_count": pause_result["pause_count"],
            "average_pause": pause_result["average_pause"],
        }
        vision_metrics_for_scoring = {
            "eye_contact_percent": eye_contact_result["eye_contact_percent"],
            "facing_camera_percent": head_pose_result["facing_camera_percent"],
        }
        scores = calculate_overall_score(
            speech_metrics_for_scoring, vision_metrics_for_scoring, llm_feedback
        )

        progress.progress(100, text="Done!")

        # Stash everything in session state so it survives Streamlit reruns
        # (e.g. when the user interacts with a widget after results are shown)
        st.session_state["results"] = {
            "question": question,
            "transcript": transcript_text,
            "filler_result": filler_result,
            "wpm": wpm,
            "pause_result": pause_result,
            "eye_contact_result": eye_contact_result,
            "head_pose_result": head_pose_result,
            "smile_result": smile_result,
            "llm_feedback": llm_feedback,
            "resume_result": resume_result,
            "scores": scores,
        }


# --- Results display ---
if "results" in st.session_state:
    r = st.session_state["results"]

    st.divider()
    st.header("Results")

    # Overall score front and center
    score_cols = st.columns(5)
    score_cols[0].metric("Overall", f"{r['scores']['overall']}/100")
    score_cols[1].metric("Confidence", f"{r['scores']['confidence']}/100")
    score_cols[2].metric("Communication", f"{r['scores']['communication']}/100")
    score_cols[3].metric("Body language", f"{r['scores']['body_language']}/100")
    score_cols[4].metric("Technical", f"{r['scores']['technical']}/100")

    st.divider()

    tab_transcript, tab_speech, tab_vision, tab_feedback, tab_resume = st.tabs(
        ["Transcript", "Speech metrics", "Body language", "AI feedback", "Resume check"]
    )

    with tab_transcript:
        st.subheader(f"Question: {r['question']}")
        st.write(r["transcript"])

    with tab_speech:
        c1, c2, c3 = st.columns(3)
        c1.metric("Speaking speed", f"{r['wpm']} WPM", wpm_feedback(r["wpm"]))
        c2.metric("Filler words", r["filler_result"]["total"])
        c3.metric("Pauses detected", r["pause_result"]["pause_count"])
        st.caption(f"Average pause length: {r['pause_result']['average_pause']}s")

        if r["filler_result"]["total"] > 0:
            st.write("Filler word breakdown:")
            breakdown = {k: v for k, v in r["filler_result"].items() if k != "total" and v > 0}
            st.bar_chart(breakdown)

    with tab_vision:
        c1, c2, c3 = st.columns(3)
        c1.metric("Eye contact", f"{r['eye_contact_result']['eye_contact_percent']}%")
        c2.metric("Facing camera", f"{r['head_pose_result']['facing_camera_percent']}%")
        c3.metric("Smile frequency", f"{r['smile_result']['smile_percent']}%")

        st.write("Head direction breakdown:")
        head_breakdown = {
            "Facing camera": r["head_pose_result"]["facing_camera_percent"],
            "Looking down": r["head_pose_result"]["looking_down_percent"],
            "Looking up": r["head_pose_result"]["looking_up_percent"],
            "Looking left": r["head_pose_result"]["looking_left_percent"],
            "Looking right": r["head_pose_result"]["looking_right_percent"],
        }
        st.bar_chart(head_breakdown)

    with tab_feedback:
        st.subheader("Strengths")
        for s in r["llm_feedback"]["strengths"]:
            st.write(f"- {s}")

        st.subheader("Areas to improve")
        for w in r["llm_feedback"]["weaknesses"]:
            st.write(f"- {w}")

        st.subheader("Suggested improved answer")
        st.info(r["llm_feedback"]["improved_answer"])

        st.subheader("Content rating")
        st.write(f"{r['llm_feedback']['rating']}/10")

    with tab_resume:
        if r["resume_result"] is None:
            st.write("Resume check was not enabled for this analysis.")
        else:
            st.subheader("Resume claims not substantiated in your answer")
            if r["resume_result"]["unsubstantiated_claims"]:
                for c in r["resume_result"]["unsubstantiated_claims"]:
                    st.write(f"- {c}")
            else:
                st.write("None — your answer covered the relevant resume claims well.")

            st.subheader("Resume claims you explained well")
            if r["resume_result"]["well_substantiated"]:
                for c in r["resume_result"]["well_substantiated"]:
                    st.write(f"- {c}")
            else:
                st.write("None identified with strong supporting evidence in your answer.")

            st.subheader("Summary")
            st.write(r["resume_result"]["summary"])

    # --- PDF export ---
    st.divider()
    if st.button("Generate PDF report"):
        pdf_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(pdf_dir, "interview_report.pdf")
        export_report_pdf(r, pdf_path)
        with open(pdf_path, "rb") as f:
            st.session_state["pdf_bytes"] = f.read()

    if "pdf_bytes" in st.session_state:
        st.download_button(
            label="Download PDF report",
            data=st.session_state["pdf_bytes"],
            file_name="interviewiq_report.pdf",
            mime="application/pdf",
        )