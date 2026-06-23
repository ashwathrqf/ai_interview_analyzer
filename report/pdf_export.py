"""Export the final interview analysis report as a downloadable PDF."""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=22, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=14, spaceBefore=16, spaceAfter=8, textColor=colors.HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        name="BodyTextCustom", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=10.5, leading=15,
    ))
    styles.add(ParagraphStyle(
        name="BulletItem", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=10.5, leading=15, leftIndent=14, bulletIndent=4,
    ))
    return styles


def _score_table(scores: dict, styles) -> Table:
    headers = ["Overall", "Confidence", "Communication", "Body language", "Technical"]
    keys = ["overall", "confidence", "communication", "body_language", "technical"]
    values = [f"{scores[k]}/100" for k in keys]

    data = [headers, values]
    table = Table(data, colWidths=[1.4 * inch] * 5)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d2d44")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("FONTSIZE", (0, 1), (-1, 1), 14),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f5f5f7")),
        ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#1a1a2e")),
    ]))
    return table


def _bullet_list(items: list, styles, empty_text: str = "None.") -> list:
    if not items:
        return [Paragraph(empty_text, styles["BodyTextCustom"])]
    return [Paragraph(f"&bull;&nbsp;&nbsp;{item}", styles["BulletItem"]) for item in items]


def export_report_pdf(report_data: dict, output_path: str) -> str:
    """Generate a PDF report from report_data and save to output_path.

    Args:
        report_data: dict with keys matching the Streamlit app's session
            state structure: "question", "transcript", "filler_result",
            "wpm", "pause_result", "eye_contact_result", "head_pose_result",
            "smile_result", "llm_feedback", "resume_result", "scores"
        output_path: where to save the generated PDF

    Returns:
        output_path on success.
    """
    styles = _build_styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )
    elements = []

    # --- Title ---
    elements.append(Paragraph("InterviewIQ AI &mdash; Performance Report", styles["ReportTitle"]))
    elements.append(Paragraph(
        "Multimodal mock interview analysis: speech, body language, and content feedback",
        styles["BodyTextCustom"],
    ))
    elements.append(Spacer(1, 16))

    # --- Scores ---
    elements.append(_score_table(report_data["scores"], styles))
    elements.append(Spacer(1, 20))

    # --- Question and transcript ---
    elements.append(Paragraph(f"Question: {report_data['question']}", styles["SectionHeading"]))
    elements.append(Paragraph(report_data["transcript"], styles["BodyTextCustom"]))

    # --- Speech metrics ---
    elements.append(Paragraph("Speech metrics", styles["SectionHeading"]))
    filler_total = report_data["filler_result"]["total"]
    pause_count = report_data["pause_result"]["pause_count"]
    avg_pause = report_data["pause_result"]["average_pause"]
    elements.append(Paragraph(
        f"Speaking speed: {report_data['wpm']} WPM &nbsp;|&nbsp; "
        f"Filler words: {filler_total} &nbsp;|&nbsp; "
        f"Pauses detected: {pause_count} (avg {avg_pause}s)",
        styles["BodyTextCustom"],
    ))

    # --- Body language ---
    elements.append(Paragraph("Body language", styles["SectionHeading"]))
    eye_contact = report_data["eye_contact_result"]["eye_contact_percent"]
    facing_camera = report_data["head_pose_result"]["facing_camera_percent"]
    smile = report_data["smile_result"]["smile_percent"]
    elements.append(Paragraph(
        f"Eye contact: {eye_contact}% &nbsp;|&nbsp; "
        f"Facing camera: {facing_camera}% &nbsp;|&nbsp; "
        f"Smile frequency: {smile}%",
        styles["BodyTextCustom"],
    ))

    # --- AI feedback ---
    elements.append(Paragraph("AI feedback on your answer", styles["SectionHeading"]))
    llm = report_data["llm_feedback"]

    elements.append(Paragraph("Strengths", styles["Heading3"]))
    elements.extend(_bullet_list(llm["strengths"], styles))

    elements.append(Paragraph("Areas to improve", styles["Heading3"]))
    elements.extend(_bullet_list(llm["weaknesses"], styles))

    elements.append(Paragraph("Suggested improved answer", styles["Heading3"]))
    elements.append(Paragraph(llm["improved_answer"], styles["BodyTextCustom"]))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"Content rating: {llm['rating']}/10", styles["BodyTextCustom"]))

    # --- Resume check (only if it was run) ---
    resume_result = report_data.get("resume_result")
    if resume_result is not None:
        elements.append(PageBreak())
        elements.append(Paragraph("Resume check", styles["SectionHeading"]))

        elements.append(Paragraph("Resume claims not substantiated in your answer", styles["Heading3"]))
        elements.extend(_bullet_list(
            resume_result["unsubstantiated_claims"], styles,
            empty_text="None &mdash; your answer covered the relevant resume claims well.",
        ))

        elements.append(Paragraph("Resume claims you explained well", styles["Heading3"]))
        elements.extend(_bullet_list(
            resume_result["well_substantiated"], styles,
            empty_text="None identified with strong supporting evidence in your answer.",
        ))

        elements.append(Paragraph("Summary", styles["Heading3"]))
        elements.append(Paragraph(resume_result["summary"], styles["BodyTextCustom"]))

    doc.build(elements)
    return output_path


if __name__ == "__main__":
    # Quick manual test using realistic sample data matching the session
    # state structure produced by streamlit_app.py.
    sample_report_data = {
        "question": "Tell me about yourself.",
        "transcript": (
            "Hi, I'm Ashwath. I'm studying Mechanical Engineering in IIT Madras. "
            "I live in Bangalore, my favorite hobbies are reading books, going "
            "through YouTube stuff, any new challenges and creating projects "
            "like this. I love doing stuff like this. I want to get into "
            "quantitative finance in NFO or machine learning and data analytics. "
            "And my priority of the night I may say is my honesty on my talent "
            "to lead a team and to work under pressure. Thank you."
        ),
        "filler_result": {"like": 2, "total": 2},
        "wpm": 93.3,
        "pause_result": {"pause_count": 2, "average_pause": 1.76},
        "eye_contact_result": {"eye_contact_percent": 81.8},
        "head_pose_result": {"facing_camera_percent": 81.8},
        "smile_result": {"smile_percent": 1.5},
        "llm_feedback": {
            "strengths": [
                "Clear academic background stated.",
                "Genuine enthusiasm for the field expressed.",
            ],
            "weaknesses": [
                "Answer lacked structure.",
                "No connection drawn between mechanical engineering and quant finance.",
            ],
            "improved_answer": "I'm Ashwath, a Mechanical Engineering student at IIT Madras...",
            "rating": 6,
        },
        "resume_result": {
            "unsubstantiated_claims": [
                "Engineered an end-to-end ETL pipeline using yfinance and Pandas",
                "Built a high-performance backtesting simulation engine",
            ],
            "well_substantiated": [],
            "summary": "The spoken answer did not draw on specific resume claims.",
        },
        "scores": {
            "overall": 73.1,
            "confidence": 79.7,
            "communication": 75.4,
            "body_language": 81.8,
            "technical": 60.0,
        },
    }

    output_path = export_report_pdf(sample_report_data, "data/sample_videos/test_report.pdf")
    print(f"PDF report generated at: {output_path}")