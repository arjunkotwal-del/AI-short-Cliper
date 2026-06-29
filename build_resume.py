"""Build Arjun Kotwal's resume PDF using reportlab."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

OUTPUT = r"C:\Users\user\OneDrive\Документы\Resumes\Base Resume\Arjun_Kotwal_resume_updated.pdf"

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=letter,
    leftMargin=0.65*inch,
    rightMargin=0.65*inch,
    topMargin=0.45*inch,
    bottomMargin=0.45*inch,
)

W = letter[0] - 1.30*inch  # usable width

# ── Styles ────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

name_style = ParagraphStyle(
    "Name", fontSize=18, leading=22, alignment=TA_CENTER,
    fontName="Helvetica-Bold", spaceAfter=2,
)
contact_style = ParagraphStyle(
    "Contact", fontSize=9.5, leading=13, alignment=TA_CENTER,
    fontName="Helvetica", spaceAfter=4,
)
section_style = ParagraphStyle(
    "Section", fontSize=11, leading=13, fontName="Helvetica-Bold",
    spaceBefore=5, spaceAfter=1,
)
job_title_style = ParagraphStyle(
    "JobTitle", fontSize=9.5, leading=12, fontName="Helvetica-Bold",
    spaceBefore=3, spaceAfter=0,
)
italic_style = ParagraphStyle(
    "Italic", fontSize=9.5, leading=12, fontName="Helvetica-Oblique",
    spaceBefore=0, spaceAfter=1,
)
body_style = ParagraphStyle(
    "Body", fontSize=9.5, leading=12, fontName="Helvetica",
    spaceBefore=0, spaceAfter=1,
)
bullet_style = ParagraphStyle(
    "Bullet", fontSize=9.5, leading=12, fontName="Helvetica",
    leftIndent=12, firstLineIndent=-12, spaceBefore=1, spaceAfter=1,
)
skills_style = ParagraphStyle(
    "Skills", fontSize=9.5, leading=12, fontName="Helvetica",
    spaceBefore=1, spaceAfter=1,
)

def hr():
    return HRFlowable(width="100%", thickness=0.8, color=colors.black, spaceAfter=3, spaceBefore=1)

def section(title):
    return [Paragraph(title, section_style), hr()]

def bullet(text):
    return Paragraph(f"•  {text}", bullet_style)

def skill_line(label, value):
    return Paragraph(f"<b>{label}</b> {value}", skills_style)

def proj_header(title, sub):
    return [
        Paragraph(f"<b>{title}</b>", job_title_style),
        Paragraph(sub, italic_style),
    ]

# ── Right-aligned date helper (two-column table) ──────────────────────────────
def two_col(left_para, right_text):
    right_para = Paragraph(right_text, ParagraphStyle(
        "Right", fontSize=9.5, leading=13, fontName="Helvetica",
        alignment=TA_RIGHT,
    ))
    t = Table([[left_para, right_para]], colWidths=[W*0.72, W*0.28])
    t.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING",   (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0), (-1,-1), 0),
    ]))
    return t

# ── Build story ───────────────────────────────────────────────────────────────
story = []

# Name + contact
story.append(Paragraph("Arjun Kotwal", name_style))
story.append(Paragraph(
    "Toronto, ON — (647) 463–1610 — arjunkotwal1607@gmail.com — GitHub — Portfolio",
    contact_style,
))

# ── Education ─────────────────────────────────────────────────────────────────
story += section("Education")
story.append(two_col(
    Paragraph("<b>Bachelor of Science (BSc), Computer Science (Honours)</b>", job_title_style),
    "Oshawa, ON",
))
story.append(two_col(
    Paragraph("Ontario Tech University", body_style),
    "2025 – Present",
))
story.append(Paragraph(
    "Relevant Coursework: Data Structures, Programming Workshop II, Computer Architecture, "
    "Calculus II, Physics II (Mechanics &amp; Waves)",
    body_style,
))

# ── Technical Skills ──────────────────────────────────────────────────────────
story += section("Technical Skills")
story.append(skill_line("Languages:", "JavaScript (ES6+), TypeScript, Python, C++, SQL"))
story.append(skill_line("Frontend:", "React, Vue.js, HTML, CSS"))
story.append(skill_line("Backend &amp; APIs:", "FastAPI, Node.js, REST APIs"))
story.append(skill_line("Databases:", "PostgreSQL, MongoDB, Firebase, Supabase"))
story.append(skill_line("Tools &amp; Platforms:", "Git/GitHub, Docker, Vercel, Netlify, VS Code, FFmpeg, OpenAI API"))
story.append(skill_line("Development Practices:", "Agile, Iterative, Waterfall (SDLC)"))

# ── Work Experience ───────────────────────────────────────────────────────────
story += section("Work Experience")
story.append(two_col(
    Paragraph("<b>Web Developer</b> — Center of Consciousness Awareness, Scotland", job_title_style),
    "Remote",
))
story.append(two_col(
    Paragraph("<i>Contract</i>", italic_style),
    "2025",
))
for b in [
    "Designed, developed, and deployed a responsive website for a UK-based organization.",
    "Built clean UI layouts optimized for mobile, desktop, and performance.",
    "Translated stakeholder requirements into functional production features.",
    "Maintained site reliability, accessibility, and post-deployment updates.",
    "Documented features and communicated technical progress with non-technical stakeholders.",
]:
    story.append(bullet(b))

# ── Projects ──────────────────────────────────────────────────────────────────
story += section("Projects")

# Project 1
story += proj_header(
    "Internship Intelligence Platform — Full-Stack Application",
    "React, FastAPI, PostgreSQL, Supabase, LLM APIs",
)
for b in [
    "Designed and built a full-stack internship tracking platform with authentication, dashboards, and analytics.",
    "Developed RESTful APIs and normalized relational database schemas to support user-specific workflows.",
    "Implemented secure authentication, protected routes, and role-based access controls.",
    "Integrated LLM-powered features to analyze resumes and match candidates to job postings.",
    "Applied iterative development practices, testing, and debugging to ensure reliability and data consistency.",
]:
    story.append(bullet(b))

story.append(Spacer(1, 2))

# Project 2 — AI YouTube Shorts Generator
story += proj_header(
    "AI YouTube Shorts Generator — CLI Automation Tool",
    "Python, FFmpeg, OpenAI GPT-4o-mini, MediaPipe, faster-Whisper",
)
for b in [
    "Built a local CLI pipeline that converts any YouTube video into viral-ready 9:16 shorts using Python, FFmpeg, OpenAI GPT-4o-mini, and MediaPipe.",
    "Engineered a virality scoring system that evaluates every transcript segment across 8 signals (hook strength, emotional peak, opinion bomb, revelation, conflict, quotable, story peak, practical value) to rank and select the top clips.",
    "Implemented dynamic speaker framing using MediaPipe face detection — samples 1 frame/second, builds a smooth crop keyframe timeline, and pans the frame to follow the active speaker.",
    "Built a gaming highlight mode that analyzes audio waveforms to detect peak dB moments and cuts clips around high-intensity events with AI-generated hook text.",
    "Integrated faster-Whisper for local word-level transcription with intelligent caching to avoid redundant processing, with optional CUDA GPU acceleration for ~10× speedup.",
]:
    story.append(bullet(b))

# ── Additional Information ────────────────────────────────────────────────────
story += section("Additional Information")
for b in [
    "Experience working in Agile and Iterative development environments across multiple projects.",
    "Strong analytical, communication, and teamwork skills developed through academic and real-world projects.",
]:
    story.append(bullet(b))

# ── Build ─────────────────────────────────────────────────────────────────────
doc.build(story)
print(f"Saved: {OUTPUT}")
