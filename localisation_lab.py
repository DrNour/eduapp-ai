import time
import streamlit as st

from translation_lab import (
    load_json,
    save_json,
    SUBMISSIONS_FILE,
    update_leaderboard,
    evaluate_translation,
    show_leaderboard,
)
from feedback_core import quick_linguistic_hints, generate_feedback
from ai_feedback import ask_ai_tutor


LOC_TASKS = {
    "LOC_1": {
        "title": "Translation vs Localisation",
        "source": "Summer Sale! Buy 1 Get 1 Free. Limited time only.",
        "instructions": "Localise this retail message for an Arabic-speaking audience. Explain at least one choice.",
    },
    "LOC_2": {
        "title": "Cultural Adaptation in Advertising",
        "source": "Grab your pumpkin spice latte and enjoy the fall vibes.",
        "instructions": "Adapt the message culturally instead of translating word for word.",
    },
    "LOC_3": {
        "title": "Conventions: Dates, Units, Currency",
        "source": "Offer valid until 12/05/2026. Price: $19.99. Weight: 3 lbs.",
        "instructions": "Rewrite this for Arabic users with appropriate local conventions.",
    },
    "LOC_4": {
        "title": "Tone and UX Microcopy",
        "source": "Oops! Something went wrong. Please try again later.",
        "instructions": "Localise this interface message so it sounds natural and user-friendly in Arabic.",
    },
    "LOC_5": {
        "title": "Post-editing: Error Detection",
        "source": "Welcome back! Continue your learning journey today.",
        "instructions": "Fix the weak localisation and explain what was wrong.",
    },
    "LOC_6": {
        "title": "App Store Description",
        "source": "Track your fitness goals, monitor progress, and stay motivated every day.",
        "instructions": "Produce a polished localised app-store style description in Arabic.",
    },
    "LOC_7": {
        "title": "Strategy and Theory Reflection",
        "source": "A global brand wants its website to feel local in every market.",
        "instructions": "Write a short reflection on which localisation strategies you would use and why.",
    },
}


def localisation_lab():
    st.title("🌍 Localisation Lab")
    st.write("Interactive English–Arabic localisation practice with saving, feedback, and optional AI tutor help.")

    student_name = st.text_input("Enter your name")
    if not student_name:
        return

    submissions = load_json(SUBMISSIONS_FILE)
    if student_name not in submissions:
        submissions[student_name] = {}

    task_id = st.sidebar.selectbox(
        "Choose a localisation exercise",
        list(LOC_TASKS.keys()),
        format_func=lambda x: f"{x} – {LOC_TASKS[x]['title']}",
    )
    task = LOC_TASKS[task_id]

    start_key = f"loc_start_{student_name}_{task_id}"
    if start_key not in st.session_state:
        st.session_state[start_key] = time.time()

    st.subheader(task["title"])
    st.write(task["instructions"])
    st.markdown("**Source text**")
    st.write(task["source"])

    with st.form(f"loc_form_{task_id}"):
        main_text = st.text_area("Your localised version", height=220)
        reflection = st.text_area("Brief reflection on your choices", height=100)
        submitted = st.form_submit_button("Submit localisation task")

    if not submitted:
        return

    time_spent = time.time() - st.session_state[start_key]
    metrics = evaluate_translation(
        main_text,
        mt_text=None,
        reference=None,
        task_type="Localisation",
        source_text=task["source"],
    )

    submissions[student_name][task_id] = {
        "source_text": task["source"],
        "mt_text": None,
        "student_text": main_text,
        "task_type": "Localisation",
        "time_spent_sec": round(time_spent, 2),
        "keystrokes": len(main_text),
        "metrics": metrics,
        "reflection": reflection,
    }
    save_json(SUBMISSIONS_FILE, submissions)
    update_leaderboard(student_name, 20)

    st.success("Localisation task saved.")
    extra = quick_linguistic_hints(task["source"], main_text)
    feedback = generate_feedback(metrics, "Localisation", extra)

    st.subheader("Adaptive Feedback")
    if feedback:
        for item in feedback:
            st.markdown(item)
    else:
        st.info("No specific issue was triggered. Focus on clarity, tone, and audience fit.")

    st.subheader("AI Tutor Chat")
    student_prompt = st.text_area(
        "Ask the AI tutor about this localisation task",
        height=120,
        key=f"loc_ai_prompt_{task_id}",
    )
    if st.button("Ask AI Tutor", key=f"loc_ai_btn_{task_id}"):
        if not student_prompt.strip():
            st.warning("Please write a prompt first.")
        else:
            with st.spinner("Requesting AI response..."):
                ai_text = ask_ai_tutor(
                    source_text=task["source"],
                    mt_text="",
                    student_text=main_text,
                    task_type="Localisation",
                    student_prompt=student_prompt,
                )
            if ai_text:
                st.markdown("### AI Tutor Response")
                st.write(ai_text)
            else:
                st.warning("AI is not configured. Add OPENAI_API_KEY to your Streamlit secrets or environment.")

    show_leaderboard()
