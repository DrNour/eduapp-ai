import os
from typing import Optional

import streamlit as st

try:
    import openai
except Exception:
    openai = None


SYSTEM_TUTOR_PROMPT = """
You are an expert English-Arabic translation tutor for university students.
Help students learn rather than simply giving final answers.
Prioritize accuracy, register, idiomaticity, cultural appropriateness, and concise pedagogy.
Do not complete the entire assignment unless the student explicitly asks for a full sample.
Use English and Arabic when helpful.
""".strip()


def _get_secret(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def get_ai_backend_status():
    openai_key = _get_secret("OPENAI_API_KEY", "")
    hf_token = _get_secret("HF_API_TOKEN", "")
    openai_model = _get_secret("OPENAI_MODEL", "gpt-4.1-mini")
    return {
        "openai": bool(openai_key),
        "openai_model": openai_model if openai_key else None,
        "hf": bool(hf_token),
    }


def build_ai_feedback_prompt(source_text: str, mt_text: str, student_text: str, task_type: str) -> str:
    mt_block = mt_text if mt_text else "(no MT output – direct translation task)"
    return f"""
You are an expert English–Arabic translation trainer specialising in translation and MT post-editing.

TASK TYPE: {task_type}

SOURCE TEXT:
{source_text}

MT OUTPUT (if any):
{mt_block}

STUDENT VERSION:
{student_text}

Give concise feedback suitable for a university translation classroom. Please:
1) Comment on accuracy (meaning transfer).
2) Comment on register and appropriateness for the context.
3) Comment on idiomatic and culturally appropriate choices.
4) Highlight one or two concrete examples where the student could improve.
5) If useful, propose a short improved version of one or two sentences, not the entire text.

You may answer partly in Arabic where it helps the student, but keep the structure clear and concise.
""".strip()


def _openai_responses_call(system_prompt: str, user_prompt: str, model: str) -> Optional[str]:
    if not openai or not hasattr(openai, "OpenAI"):
        return None
    api_key = _get_secret("OPENAI_API_KEY", "")
    if not api_key:
        return None

    client = openai.OpenAI(api_key=api_key)
    resp = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
        max_output_tokens=700,
    )
    text = getattr(resp, "output_text", None)
    if text:
        return text.strip()
    return None


def _openai_chat_call(system_prompt: str, user_prompt: str, model: str) -> Optional[str]:
    if not openai:
        return None
    api_key = _get_secret("OPENAI_API_KEY", "")
    if not api_key:
        return None

    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=700,
            temperature=0.4,
        )
        text = resp.choices[0].message.content
        return text.strip() if text else None

    openai.api_key = api_key
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=700,
        temperature=0.4,
    )
    text = resp.choices[0].message["content"]
    return text.strip() if text else None


def generate_ai_feedback(prompt: str) -> Optional[str]:
    model = _get_secret("OPENAI_MODEL", "gpt-4.1-mini")
    try:
        text = _openai_responses_call(
            "You are a helpful, expert translation instructor.", prompt, model
        )
        if text:
            return text
    except Exception:
        pass

    try:
        text = _openai_chat_call(
            "You are a helpful, expert translation instructor.", prompt, model
        )
        if text:
            return text
    except Exception:
        pass

    return None


def ask_ai_tutor(
    source_text: str,
    mt_text: str,
    student_text: str,
    task_type: str,
    student_prompt: str,
) -> Optional[str]:
    model = _get_secret("OPENAI_MODEL", "gpt-4.1-mini")
    user_prompt = f"""
TASK TYPE: {task_type}

SOURCE TEXT:
{source_text}

MT OUTPUT:
{mt_text or '(none)'}

STUDENT SUBMISSION:
{student_text}

STUDENT QUESTION:
{student_prompt}
""".strip()

    try:
        text = _openai_responses_call(SYSTEM_TUTOR_PROMPT, user_prompt, model)
        if text:
            return text
    except Exception:
        pass

    try:
        text = _openai_chat_call(SYSTEM_TUTOR_PROMPT, user_prompt, model)
        if text:
            return text
    except Exception:
        pass

    return None
