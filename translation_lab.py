import os
import re
import json
import time
import hashlib
import threading
from io import BytesIO
from pathlib import Path
import datetime

import streamlit as st
import pandas as pd
from docx import Document
from docx.shared import RGBColor
from difflib import ndiff

try:
    import sacrebleu
except Exception:
    sacrebleu = None

try:
    from bert_score import score as bertscore_score
except Exception:
    bertscore_score = None

from feedback_core import tokenize, compute_edit_details, quick_linguistic_hints, generate_feedback
from ai_feedback import build_ai_feedback_prompt, generate_ai_feedback, ask_ai_tutor, get_ai_backend_status


try:
    THIS_FILE = os.path.abspath(__file__)
    LAST_EDIT = datetime.datetime.fromtimestamp(os.path.getmtime(THIS_FILE))
except Exception:
    THIS_FILE = "interactive_session"
    LAST_EDIT = datetime.datetime.now()

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

EXERCISES_FILE = DATA_DIR / "exercises.json"
SUBMISSIONS_FILE = DATA_DIR / "submissions.json"
LEADERBOARD_FILE = DATA_DIR / "leaderboard.json"

_lock = threading.Lock()
_INVALID_XML_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def load_json(file: Path):
    file = Path(file)
    if file.exists():
        with file.open("r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def save_json(file: Path, data):
    with _lock:
        tmp = Path(str(file) + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        tmp.replace(file)


def _env(name, default=""):
    return os.getenv(name, default)


_INSTRUCTOR_PLAIN = _env("INSTRUCTOR_PASSWORD_PLAIN", "")
_INSTRUCTOR_SHA256 = _env("INSTRUCTOR_PASSWORD_SHA256", "")
_INSTRUCTOR_DEV_MODE = _env("INSTRUCTOR_DEV_MODE", "0") == "1"
_FALLBACK_PLAIN = "admin123"


def check_password(typed: str) -> bool:
    try:
        if _INSTRUCTOR_SHA256:
            return hashlib.sha256(typed.encode("utf-8")).hexdigest() == _INSTRUCTOR_SHA256
        if _INSTRUCTOR_PLAIN:
            return typed == _INSTRUCTOR_PLAIN
        if _INSTRUCTOR_DEV_MODE:
            return typed == _FALLBACK_PLAIN
        return False
    except Exception:
        return False


def evaluate_translation(student_text, mt_text=None, reference=None, task_type="Translate", source_text=""):
    src_len = max(1, len(tokenize(source_text)))
    tgt_len = len(tokenize(student_text))
    length_ratio = round(tgt_len / src_len, 3)

    if task_type == "Post-edit MT" and mt_text:
        additions, deletions, edits = compute_edit_details(mt_text, student_text)
    else:
        additions = deletions = edits = 0

    bleu = chrf = bert_f1 = None
    if reference:
        refs = [reference]
        try:
            if sacrebleu:
                bleu = float(sacrebleu.corpus_bleu([student_text], [refs]).score)
                chrf = float(sacrebleu.corpus_chrf([student_text], [refs]).score)
        except Exception:
            bleu = None
            chrf = None
        try:
            if bertscore_score:
                _, _, f1 = bertscore_score([student_text], [reference], lang="en")
                bert_f1 = float(f1.mean().item())
        except Exception:
            bert_f1 = None

    return {
        "length_ratio": length_ratio,
        "BLEU": None if bleu is None else round(bleu, 2),
        "chrF++": None if chrf is None else round(chrf, 2),
        "BERTScore_F1": None if bert_f1 is None else round(bert_f1, 3),
        "additions": additions,
        "deletions": deletions,
        "edits": edits,
    }


def _join_tokens_for_display(tokens):
    out = " ".join(tokens)
    out = re.sub(r"\s+([.,!?;:])", r"\1", out)
    return out


def diff_text(baseline: str, student_text: str) -> str:
    differ = ndiff(tokenize(baseline), tokenize(student_text))
    parts = []
    for w in differ:
        token = w[2:]
        if w.startswith("- "):
            parts.append(f"<span style='color:#c00;text-decoration:line-through'>{token}</span>")
        elif w.startswith("+ "):
            parts.append(f"<span style='color:#080'>{token}</span>")
        else:
            parts.append(token)
    return _join_tokens_for_display(parts)


def _safe_docx_text(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return _INVALID_XML_RE.sub("", value)


def add_diff_to_doc(doc: Document, baseline: str, student_text: str):
    differ = ndiff(tokenize(baseline), tokenize(student_text))
    p = doc.add_paragraph()
    for w in differ:
        token = _safe_docx_text(w[2:])
        if w.startswith("- "):
            run = p.add_run(token + " ")
            run.font.strike = True
            run.font.color.rgb = RGBColor(255, 0, 0)
        elif w.startswith("+ "):
            run = p.add_run(token + " ")
            run.font.color.rgb = RGBColor(0, 128, 0)
        else:
            p.add_run(token + " ")


def export_student_word(submissions, student_name):
    doc = Document()
    doc.add_heading(_safe_docx_text(f"Student: {student_name}"), 0)
    subs = submissions.get(student_name, {})
    for ex_id, sub in subs.items():
        doc.add_heading(_safe_docx_text(f"Exercise {ex_id}"), level=1)
        doc.add_paragraph("Source Text:")
        doc.add_paragraph(_safe_docx_text(sub.get("source_text", "")))
        if sub.get("mt_text"):
            doc.add_paragraph("MT Output:")
            doc.add_paragraph(_safe_docx_text(sub.get("mt_text", "")))

        if sub.get("task_type") == "Post-edit MT":
            doc.add_paragraph("Student Submission (Track Changes):")
            add_diff_to_doc(doc, sub.get("mt_text", "") or "", sub.get("student_text", ""))
        else:
            doc.add_paragraph("Student Submission:")
            doc.add_paragraph(_safe_docx_text(sub.get("student_text", "")))

        doc.add_paragraph(_safe_docx_text(f"Metrics: {sub.get('metrics', {})}"))
        doc.add_paragraph(_safe_docx_text(f"Task Type: {sub.get('task_type', '')}"))
        doc.add_paragraph(_safe_docx_text(f"Time Spent: {sub.get('time_spent_sec', 0):.2f} sec"))
        doc.add_paragraph(_safe_docx_text(f"Characters Typed: {sub.get('keystrokes', 0)}"))
        if sub.get("reflection"):
            doc.add_paragraph("Reflection:")
            doc.add_paragraph(_safe_docx_text(sub.get("reflection")))
        doc.add_paragraph("---")

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


def export_summary_excel(submissions):
    rows = []
    for student, subs in submissions.items():
        for ex_id, sub in subs.items():
            m = sub.get("metrics", {})
            rows.append(
                {
                    "Student": student,
                    "Exercise": ex_id,
                    "Task Type": sub.get("task_type", ""),
                    "Length Ratio": m.get("length_ratio"),
                    "BLEU": m.get("BLEU"),
                    "chrF++": m.get("chrF++"),
                    "BERTScore_F1": m.get("BERTScore_F1"),
                    "Additions": m.get("additions"),
                    "Deletions": m.get("deletions"),
                    "Edits": m.get("edits"),
                    "Time Spent (s)": sub.get("time_spent_sec", 0),
                    "Characters Typed": sub.get("keystrokes", 0),
                }
            )
    df = pd.DataFrame(rows)
    buf = BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf


def load_leaderboard():
    return load_json(LEADERBOARD_FILE)


def update_leaderboard(student_name, points):
    leaderboard = load_leaderboard()
    leaderboard[student_name] = leaderboard.get(student_name, 0) + points
    save_json(LEADERBOARD_FILE, leaderboard)


def show_leaderboard():
    leaderboard = load_leaderboard()
    st.subheader("Leaderboard")
    if leaderboard:
        items = sorted(leaderboard.items(), key=lambda x: x[1], reverse=True)
        st.dataframe(pd.DataFrame(items, columns=["Student", "Points"]), use_container_width=True)
    else:
        st.info("No leaderboard data yet.")


def instructor_dashboard():
    st.title("Instructor Dashboard")
    password = st.text_input("Enter instructor password", type="password")
    if not check_password(password):
        if not (_INSTRUCTOR_PLAIN or _INSTRUCTOR_SHA256 or _INSTRUCTOR_DEV_MODE):
            st.warning("Instructor password is not configured. Set INSTRUCTOR_PASSWORD_PLAIN or INSTRUCTOR_PASSWORD_SHA256.")
        else:
            st.warning("Incorrect password. Access denied.")
        return

    st.sidebar.info(f"Loaded: {THIS_FILE}\n\nLast modified: {LAST_EDIT:%Y-%m-%d %H:%M:%S}")

    status = get_ai_backend_status()
    st.subheader("AI Diagnostics")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**OpenAI**")
        st.write(f"✅ Available ({status['openai_model']})" if status["openai"] else "❌ Not configured")
    with col2:
        st.write("**Hugging Face**")
        st.write("✅ Token found" if status["hf"] else "❌ Not configured")

    exercises = load_json(EXERCISES_FILE)
    submissions = load_json(SUBMISSIONS_FILE)

    st.subheader("Create or Edit Exercise")
    selected_ex = st.selectbox("Select Exercise", ["New"] + list(exercises.keys()))

    default_source = exercises.get(selected_ex, {}).get("source_text", "") if selected_ex != "New" else ""
    default_mt = exercises.get(selected_ex, {}).get("mt_text", "") if selected_ex != "New" else ""

    with st.form("exercise_form"):
        source_text = st.text_area("Source Text", value=default_source, height=150)
        mt_text = st.text_area("MT Output (optional)", value=default_mt or "", height=150)
        c1, c2 = st.columns(2)
        save_btn = c1.form_submit_button("Save Exercise")
        delete_btn = c2.form_submit_button("Delete Exercise")

    if save_btn:
        try:
            next_id = str(max([int(k) for k in exercises.keys()] + [0]) + 1).zfill(3) if selected_ex == "New" else selected_ex
        except Exception:
            next_id = "001" if selected_ex == "New" else selected_ex
        exercises[next_id] = {"source_text": source_text, "mt_text": mt_text.strip() or None}
        save_json(EXERCISES_FILE, exercises)
        st.success(f"Exercise saved: {next_id}")

    if delete_btn and selected_ex != "New":
        exercises.pop(selected_ex, None)
        save_json(EXERCISES_FILE, exercises)
        st.success(f"Exercise {selected_ex} deleted.")

    st.subheader("Downloads")
    if submissions:
        student_choice = st.selectbox("Choose student", ["All"] + list(submissions.keys()))
        if student_choice != "All":
            buf = export_student_word(submissions, student_choice)
            st.download_button(
                f"Download {student_choice} report",
                buf,
                file_name=f"{re.sub(r'[^\w\-]+', '_', student_choice)}_submissions.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        st.download_button(
            "Download Excel Summary",
            export_summary_excel(submissions),
            file_name="metrics_summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        show_leaderboard()
    else:
        st.info("No submissions yet.")


def student_dashboard():
    st.title("Student Dashboard")

    exercises = load_json(EXERCISES_FILE)
    if not exercises:
        st.info("No exercises available yet.")
        return

    submissions = load_json(SUBMISSIONS_FILE)
    student_name = st.text_input("Enter your name")
    if not student_name:
        return

    if student_name not in submissions:
        submissions[student_name] = {}

    ex_id = st.selectbox("Choose Exercise", list(exercises.keys()))
    ex = exercises[ex_id]

    st.subheader("Source Text")
    st.markdown(
        f"<div style='font-family:Times New Roman;font-size:12pt;'>{ex.get('source_text', '')}</div>",
        unsafe_allow_html=True,
    )

    task_options = ["Translate"] if not ex.get("mt_text") else ["Translate", "Post-edit MT"]
    task_type = st.radio("Task Type", task_options, horizontal=True)
    initial_text = "" if task_type == "Translate" else (ex.get("mt_text", "") or "")

    start_key = f"start_time_{student_name}_{ex_id}"
    keys_key = f"chars_{student_name}_{ex_id}"
    if start_key not in st.session_state:
        st.session_state[start_key] = time.time()
    if keys_key not in st.session_state:
        st.session_state[keys_key] = 0

    with st.form(key=f"exercise_form_{student_name}_{ex_id}"):
        student_text = st.text_area("Type your translation or post-edit here", initial_text, height=300)
        reflection = st.text_area("Brief reflection", "", height=80)
        submitted = st.form_submit_button("Submit")

    if not submitted:
        return

    time_spent = time.time() - st.session_state[start_key]
    st.session_state[keys_key] = len(student_text)

    metrics = evaluate_translation(
        student_text,
        mt_text=ex.get("mt_text"),
        reference=None,
        task_type=task_type,
        source_text=ex.get("source_text", ""),
    )

    submissions[student_name][ex_id] = {
        "source_text": ex.get("source_text", ""),
        "mt_text": ex.get("mt_text"),
        "student_text": student_text,
        "task_type": task_type,
        "time_spent_sec": round(time_spent, 2),
        "keystrokes": st.session_state[keys_key],
        "metrics": metrics,
        "reflection": reflection,
    }
    save_json(SUBMISSIONS_FILE, submissions)

    points = 0
    if metrics.get("BLEU") is not None:
        points += int(metrics["BLEU"])
    if metrics.get("chrF++") is not None:
        points += int(metrics["chrF++"] / 2)
    if task_type == "Post-edit MT":
        points += max(0, 10 - int(metrics["edits"]))
    update_leaderboard(student_name, points)

    st.success("Submission saved.")

    def _fmt(v):
        return "—" if v is None else v

    st.subheader("Your Metrics")
    st.markdown(f"""
- **Length Ratio**: {_fmt(metrics['length_ratio'])}
- **BLEU**: {_fmt(metrics['BLEU'])}
- **chrF++**: {_fmt(metrics['chrF++'])}
- **BERTScore F1**: {_fmt(metrics['BERTScore_F1'])}
- **Additions**: {_fmt(metrics['additions'])}
- **Deletions**: {_fmt(metrics['deletions'])}
- **Edits**: {_fmt(metrics['edits'])}
- **Time Spent**: {round(time_spent, 2)} sec
- **Characters Typed**: {st.session_state[keys_key]}
""")

    extra = quick_linguistic_hints(ex.get("source_text", ""), student_text)
    feedback_msgs = generate_feedback(metrics, task_type, extra)
    st.subheader("Adaptive Feedback")
    if feedback_msgs:
        for msg in feedback_msgs:
            st.markdown(msg)
    else:
        st.info("No specific issues triggered. Focus on cohesion, clarity, and consistent terminology.")

    if task_type == "Post-edit MT":
        st.subheader("Track Changes")
        st.caption("Green = additions, red strike = deletions.")
        st.markdown(diff_text(ex.get("mt_text", "") or "", student_text), unsafe_allow_html=True)

    history = []
    for prev_ex_id, prev_sub in submissions.get(student_name, {}).items():
        prev_metrics = prev_sub.get("metrics", {})
        history.append({
            "ex": prev_ex_id,
            "BLEU": prev_metrics.get("BLEU"),
            "chrF++": prev_metrics.get("chrF++"),
            "Edits": prev_metrics.get("edits", 0),
        })
    if history:
        st.subheader("Progress Overview")
        df_hist = pd.DataFrame(history).set_index("ex")
        cols = [c for c in ["BLEU", "chrF++"] if c in df_hist.columns]
        if cols:
            st.line_chart(df_hist[cols])
        if "Edits" in df_hist.columns:
            st.bar_chart(df_hist[["Edits"]])

    show_leaderboard()

    st.subheader("Optional AI Feedback")
    if st.button("Get AI feedback on your submission"):
        prompt = build_ai_feedback_prompt(
            ex.get("source_text", ""),
            ex.get("mt_text", "") or "",
            student_text,
            task_type,
        )
        with st.spinner("Requesting AI feedback..."):
            ai_text = generate_ai_feedback(prompt)
        if ai_text:
            st.markdown("### AI Feedback")
            st.write(ai_text)
        else:
            st.warning("AI feedback is not configured. Add OPENAI_API_KEY and optionally OPENAI_MODEL.")

    st.subheader("AI Tutor Chat")
    student_prompt = st.text_area(
        "Write your own prompt to the AI tutor",
        height=120,
        placeholder="Example: Explain whether my Arabic sounds too literal and suggest two more idiomatic alternatives.",
    )
    if st.button("Ask AI Tutor"):
        if not student_prompt.strip():
            st.warning("Please write a prompt first.")
        else:
            with st.spinner("Requesting AI response..."):
                ai_text = ask_ai_tutor(
                    source_text=ex.get("source_text", ""),
                    mt_text=ex.get("mt_text", "") or "",
                    student_text=student_text,
                    task_type=task_type,
                    student_prompt=student_prompt,
                )
            if ai_text:
                st.markdown("### AI Tutor Response")
                st.write(ai_text)
            else:
                st.warning("AI tutor is not configured. Add OPENAI_API_KEY to Streamlit secrets or environment variables.")
