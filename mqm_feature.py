import streamlit as st
import json
import datetime
import re

from pathlib import Path

DATA_DIR = Path("./data")
MQM_FILE = DATA_DIR / "mqm_assessments.json"
MQM_CONFIG_FILE = DATA_DIR / "mqm_config.json"


# ---------------- Storage ----------------
def load_json(file):
    if file.exists():
        try:
            return json.loads(file.read_text(encoding="utf-8"))
        except:
            return {}
    return {}


def save_json(file, data):
    file.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


# ---------------- MQM Config ----------------
def load_mqm_config():
    cfg = load_json(MQM_CONFIG_FILE)
    if not cfg:
        cfg = {
            "categories": [
                "Accuracy",
                "Fluency",
                "Terminology",
                "Style",
                "Grammar",
                "Spelling / punctuation",
                "Omission",
                "Addition",
                "Mistranslation"
            ],
            "severity_weights": {
                "minor": 1,
                "major": 5,
                "critical": 10
            }
        }
        save_json(MQM_CONFIG_FILE, cfg)
    return cfg


# ---------------- MQM Storage ----------------
def load_mqm():
    data = load_json(MQM_FILE)
    return data if isinstance(data, dict) else {}


def save_mqm(student, ex_id, payload):
    data = load_mqm()
    if student not in data:
        data[student] = {}
    data[student][ex_id] = payload
    save_json(MQM_FILE, data)


def get_mqm(student, ex_id):
    return load_mqm().get(student, {}).get(ex_id, {})


# ---------------- MQM Scoring ----------------
def compute_score(errors, weights):
    penalty = 0
    for e in errors:
        penalty += weights.get(e["severity"], 0)

    return {
        "score": max(0, 100 - penalty),
        "penalty": penalty,
        "count": len(errors)
    }


# ---------------- AI MQM ----------------
def build_prompt(src, mt, student, categories):
    return f"""
You are an MQM evaluator.

Categories: {", ".join(categories)}
Severity: minor, major, critical

Return ONLY JSON list:
[{{"category":"","severity":"","span":"","comment":""}}]

SOURCE:
{src}

MT:
{mt or "(none)"}

STUDENT:
{student}
"""


def parse_ai(text, categories):
    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            data = json.loads(text)

        clean = []
        for e in data:
            if e.get("category") in categories:
                clean.append(e)
        return clean
    except:
        return []


# ---------------- UI ----------------
def mqm_rating_form(student, ex_id, submission, ask_ai_fn):

    st.subheader("MQM Assessment")

    cfg = load_mqm_config()
    categories = cfg["categories"]
    weights = cfg["severity_weights"]

    existing = get_mqm(student, ex_id)
    ai_key = f"mqm_ai_{student}_{ex_id}"

    # ---------- AI Button ----------
    if st.button("🤖 Generate AI MQM Suggestions"):
        prompt = build_prompt(
            submission["source_text"],
            submission.get("mt_text", ""),
            submission["student_text"],
            categories
        )

        ai_text = ask_ai_fn("MQM evaluator", prompt)
        suggestions = parse_ai(ai_text, categories)
        st.session_state[ai_key] = suggestions

    base = st.session_state.get(ai_key, existing.get("errors", []))

    errors = []

    for i in range(max(3, len(base))):
        st.markdown(f"**Error {i+1}**")

        old = base[i] if i < len(base) else {}

        cat = st.selectbox(
            f"Category {i}",
            categories,
            index=categories.index(old["category"]) if old.get("category") in categories else 0
        )

        sev = st.selectbox(
            f"Severity {i}",
            ["minor", "major", "critical"],
            index=["minor", "major", "critical"].index(old["severity"]) if old.get("severity") in ["minor","major","critical"] else 0
        )

        span = st.text_input(f"Span {i}", value=old.get("span",""))
        comment = st.text_area(f"Comment {i}", value=old.get("comment",""))

        if span or comment:
            errors.append({
                "category": cat,
                "severity": sev,
                "span": span,
                "comment": comment
            })

    overall = st.text_area("Overall comment", value=existing.get("overall_comment",""))

    if st.button("Save MQM"):
        score = compute_score(errors, weights)

        payload = {
            "errors": errors,
            "overall_comment": overall,
            "score": score["score"],
            "penalty": score["penalty"],
            "count": score["count"],
            "created": datetime.datetime.now().isoformat()
        }

        save_mqm(student, ex_id, payload)
        st.success("Saved!")

    # display
    if existing:
        st.markdown("### Current MQM")
        st.write(existing)
