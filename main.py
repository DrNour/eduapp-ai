import streamlit as st
from translation_lab import instructor_dashboard, student_dashboard
from localisation_lab import localisation_lab


def main():
    st.set_page_config(page_title="Translation Lab (EduApp)", layout="wide")

    st.sidebar.title("Navigation")
    st.markdown(
        "<div style='padding:8px;border:1px solid #ddd;border-radius:8px;background:#f7f9ff'>"
        "<b>EduApp – Build:</b> 2026-03-23 v5 (translation + localisation lab + AI tutor prompts)</div>",
        unsafe_allow_html=True,
    )

    section = st.sidebar.radio(
        "Module",
        ["Core Translation Lab", "Localisation Lab"],
        index=0,
    )

    if section == "Localisation Lab":
        localisation_lab()
        return

    role = st.sidebar.radio("Login as", ["Instructor", "Student"], index=1)
    if role == "Instructor":
        instructor_dashboard()
    else:
        student_dashboard()


if __name__ == "__main__":
    main()
