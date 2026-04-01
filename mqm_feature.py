import streamlit as st


MQM_CATEGORIES = [
    "accuracy",
    "fluency",
    "terminology",
    "style",
    "locale"
]


MQM_WEIGHTS = {
    "minor": 1,
    "major": 5,
    "critical": 10
}


def init_mqm_structure():
    return {
        cat: {"minor": 0, "major": 0, "critical": 0}
        for cat in MQM_CATEGORIES
    }


def compute_mqm_score(mqm_dict):
    total = 0
    for cat in MQM_CATEGORIES:
        for severity, weight in MQM_WEIGHTS.items():
            total += mqm_dict[cat][severity] * weight
    return total


def mqm_label(score):
    if score <= 5:
        return "Excellent"
    elif score <= 15:
        return "Good"
    elif score <= 30:
        return "Needs Revision"
    else:
        return "Poor"


def mqm_rating_form(student_name, ex_id, submission):
    st.subheader("MQM Evaluation")

    if "mqm" not in submission:
        submission["mqm"] = init_mqm_structure()

    mqm = submission["mqm"]

    with st.form(f"mqm_form_{student_name}_{ex_id}"):

        for cat in MQM_CATEGORIES:
            st.markdown(f"### {cat.capitalize()}")

            col1, col2, col3 = st.columns(3)
            with col1:
                mqm[cat]["minor"] = st.number_input(
                    f"{cat} minor",
                    min_value=0,
                    value=mqm[cat]["minor"],
                    key=f"{cat}_minor_{student_name}_{ex_id}"
                )
            with col2:
                mqm[cat]["major"] = st.number_input(
                    f"{cat} major",
                    min_value=0,
                    value=mqm[cat]["major"],
                    key=f"{cat}_major_{student_name}_{ex_id}"
                )
            with col3:
                mqm[cat]["critical"] = st.number_input(
                    f"{cat} critical",
                    min_value=0,
                    value=mqm[cat]["critical"],
                    key=f"{cat}_critical_{student_name}_{ex_id}"
                )

        notes = st.text_area("Instructor Notes", value=submission.get("mqm_notes", ""))

        submit = st.form_submit_button("Save MQM Evaluation")

    if submit:
        submission["mqm"] = mqm
        submission["mqm_score"] = compute_mqm_score(mqm)
        submission["mqm_label"] = mqm_label(submission["mqm_score"])
        submission["mqm_notes"] = notes

        st.success("MQM evaluation saved!")
