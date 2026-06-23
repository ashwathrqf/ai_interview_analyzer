# InterviewIQ AI

**Multimodal Interview Performance Analyzer**

InterviewIQ AI is an AI-powered tool that analyzes mock interview recordings
across speech, body language, and answer quality — then generates a scored,
downloadable performance report. Unlike most mock-interview platforms that
only ask questions, InterviewIQ AI tells you *how* you answered, not just *what*.

---

## Why

Most mock interview tools stop at asking questions. They don't tell you:

- Were you speaking confidently?
- Did you maintain eye contact?
- Did you use too many filler words?
- Did you speak too fast or too slow?
- Did you actually answer the question that was asked?

InterviewIQ AI analyzes the full recording — audio and video — and combines
speech metrics, body language metrics, and an LLM evaluation of your actual
answer into one report.

---

## Features

- **Speech-to-text transcription** (Whisper)
- **Filler word detection** (um, uh, like, you know, etc.)
- **Speaking speed (WPM)** with ideal-range comparison
- **Pause detection** (frequency and average length)
- **Eye contact estimation** (MediaPipe FaceMesh)
- **Smile frequency detection**
- **Head pose tracking** (looking down/left/right)
- **LLM-based answer evaluation** — strengths, weaknesses, and an improved
  sample answer
- **Resume-aware interview evaluation** — flags claims on your resume
  (e.g. "I built an event-driven architecture") that you didn't explain when
  asked about them
- **Combined scoring**: Confidence, Communication, Body Language, Technical,
  and an Overall score
- **PDF report export**

---

## Tech Stack

| Layer | Tool |
|---|---|
| Speech-to-text | Whisper |
| Computer vision | OpenCV, MediaPipe |
| LLM evaluation | OpenAI API |
| Video/audio processing | MoviePy |
| UI | Streamlit |
| Data & charts | Pandas, NumPy, Plotly |
| PDF export | fpdf2 |

---

## Project Structure

```
interviewiq-ai/
├── audio/          # Audio extraction from video
├── speech/         # Transcription, filler words, WPM, pauses
├── vision/         # Eye contact, smile, head pose
├── llm/            # OpenAI-based answer & resume evaluation
├── report/         # Scoring logic and PDF export
├── utils/          # Shared config/constants
├── data/           # Sample videos for local testing (gitignored)
├── tests/          # Unit tests
└── streamlit_app.py
```

---

## Setup

**Requirements:** Python 3.11

```bash
git clone https://github.com/<your-username>/interviewiq-ai.git
cd interviewiq-ai

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env       # Windows: copy .env.example .env
# then open .env and add your OPENAI_API_KEY
```

---

## Run

```bash
streamlit run streamlit_app.py
```

This launches the dashboard at `http://localhost:8501`, where you can upload
an interview video and view the generated report.

---

## Running Tests

```bash
pytest
```

---

## Roadmap

**V1 (current focus)**
- Video upload → transcription → speech metrics → vision metrics →
  LLM feedback → scored report → PDF export

**Explicitly out of scope for V1**
- Live/real-time interviews
- AI avatars or voice cloning
- Webcam-based live mode
- Interview scheduling
- A backing database

Keeping V1 tightly scoped is intentional — it's meant to ship as a complete,
polished tool rather than an open-ended platform.

---

## License

MIT