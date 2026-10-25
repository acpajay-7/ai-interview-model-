"""Streamlit UI for the local AI interview evaluator.

Run with:
    streamlit run ui.py --browser.gatherUsageStats false
"""

import json
import random
from datetime import datetime

import pandas as pd
import streamlit as st

from network_guard import enable_network_guard

# Blocks every outbound connection except localhost (Ollama).
enable_network_guard()

from scorer import MODEL_NAME, evaluate_answer  # noqa: E402

st.set_page_config(page_title="AI Interview System", page_icon="🎯", layout="centered")

# ----------------------------------------------------------------------------
# Question bank (edit freely, or use the custom table in the setup screen)
# ----------------------------------------------------------------------------
QUESTION_BANK = [
    {
        "topic": "Machine Learning",
        "question": "What is overfitting in Machine Learning?",
        "reference_answer": (
            "Overfitting occurs when a model learns the training data too closely, "
            "including noise, and performs poorly on unseen data."
        ),
    },
    {
        "topic": "Machine Learning",
        "question": "What is the difference between supervised and unsupervised learning?",
        "reference_answer": (
            "Supervised learning trains on labeled data to predict known targets "
            "(classification, regression). Unsupervised learning finds structure in "
            "unlabeled data (clustering, dimensionality reduction)."
        ),
    },
    {
        "topic": "Machine Learning",
        "question": "What is the bias-variance tradeoff?",
        "reference_answer": (
            "High bias means a model is too simple and underfits; high variance means "
            "it is too sensitive to the training data and overfits. Reducing one "
            "usually increases the other, so you look for a balance that minimizes "
            "total error on unseen data."
        ),
    },
    {
        "topic": "Machine Learning",
        "question": "What is cross-validation and why is it used?",
        "reference_answer": (
            "Cross-validation splits data into several folds, training on some and "
            "validating on the rest in rotation. It gives a more reliable estimate of "
            "generalization performance and helps detect overfitting and tune models."
        ),
    },
    {
        "topic": "Machine Learning",
        "question": "What is regularization?",
        "reference_answer": (
            "Regularization adds a penalty on model complexity (such as L1 or L2 on "
            "the weights) to the loss function to discourage overfitting and improve "
            "generalization."
        ),
    },
    {
        "topic": "Python",
        "question": "What is the difference between a list and a tuple in Python?",
        "reference_answer": (
            "Lists are mutable and can be changed after creation; tuples are immutable. "
            "Tuples are hashable if their elements are, so they can be dict keys, and "
            "are often used for fixed collections."
        ),
    },
    {
        "topic": "Python",
        "question": "What is a decorator in Python?",
        "reference_answer": (
            "A decorator is a function that takes another function and returns a "
            "modified or wrapped version of it, letting you add behavior such as "
            "logging or caching without changing the original function's code."
        ),
    },
    {
        "topic": "Databases",
        "question": "What is the difference between SQL and NoSQL databases?",
        "reference_answer": (
            "SQL databases are relational with fixed schemas and use SQL, offering "
            "strong consistency (ACID). NoSQL databases are non-relational (document, "
            "key-value, graph, column) with flexible schemas and are often easier to "
            "scale horizontally."
        ),
    },
]

STATUS_LABEL = {
    "complete": "✅ Complete",
    "partially_correct": "🟡 Partially correct",
    "incorrect": "❌ Incorrect",
    "uncertain": "❓ Uncertain",
}


# ----------------------------------------------------------------------------
# Session state helpers
# ----------------------------------------------------------------------------
def init_state():
    defaults = {
        "stage": "setup",  # setup -> interview -> report
        "questions": [],
        "idx": 0,
        "results": [],
        "candidate_name": "",
        "started_at": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_interview():
    for key in ("stage", "questions", "idx", "results", "candidate_name", "started_at"):
        st.session_state.pop(key, None)
    # drop leftover answer boxes
    for key in [k for k in st.session_state if k.startswith("answer_")]:
        del st.session_state[key]
    init_state()


def verdict(score: float) -> str:
    if score >= 8:
        return "Strong"
    if score >= 6:
        return "Good"
    if score >= 4:
        return "Needs improvement"
    return "Weak"


def render_feedback(result: dict):
    c1, c2, c3 = st.columns(3)
    c1.metric("Final score", f"{result['final_score']}/10")
    c2.metric("Correctness", f"{result['correctness_score']}/10")
    c3.metric("Depth", f"{result['knowledge_depth_score']}/10")

    st.markdown(f"**Status:** {STATUS_LABEL.get(result['answer_status'], result['answer_status'])}")
    st.markdown(f"**Feedback:** {result['explanation']}")

    if result["contradictions"]:
        st.warning("**Contradictions:**\n\n" + "\n".join(f"- {c}" for c in result["contradictions"]))
    if result["unsupported_claims"]:
        st.info("**Unsupported claims:**\n\n" + "\n".join(f"- {c}" for c in result["unsupported_claims"]))

    st.caption(f"Evaluator confidence: **{result['confidence']}**. {result['confidence_reason']}")


# ----------------------------------------------------------------------------
# Screens
# ----------------------------------------------------------------------------
def screen_setup():
    st.title("🎯 AI Interview System")
    st.write("Answer technical questions and get scored by a local AI evaluator. Nothing leaves your machine.")

    st.session_state.candidate_name = st.text_input(
        "Your name", value=st.session_state.candidate_name, placeholder="e.g. Aman"
    )

    source = st.radio("Question source", ["Built-in bank", "My own questions"], horizontal=True)

    if source == "Built-in bank":
        topics = sorted({q["topic"] for q in QUESTION_BANK})
        chosen_topics = st.multiselect("Topics", topics, default=topics)
        pool = [q for q in QUESTION_BANK if q["topic"] in chosen_topics]

        if not pool:
            st.warning("Select at least one topic.")
            return

        count = st.slider(
            "Number of questions",
            min_value=1,
            max_value=len(pool),
            value=min(5, len(pool)),
        )
        shuffle = st.checkbox("Shuffle questions", value=True)

        if st.button("Start interview", type="primary", use_container_width=True):
            questions = random.sample(pool, count) if shuffle else pool[:count]
            start_interview(questions)

    else:
        st.caption("Add one row per question. The reference answer is what the evaluator compares against.")
        df = st.data_editor(
            pd.DataFrame([{"question": "", "reference_answer": ""}]),
            num_rows="dynamic",
            use_container_width=True,
            key="custom_editor",
        )
        valid = df[
            df["question"].fillna("").str.strip().astype(bool)
            & df["reference_answer"].fillna("").str.strip().astype(bool)
        ]

        if st.button("Start interview", type="primary", use_container_width=True, disabled=valid.empty):
            questions = [
                {"topic": "Custom", **row}
                for row in valid[["question", "reference_answer"]].to_dict("records")
            ]
            start_interview(questions)


def start_interview(questions: list[dict]):
    st.session_state.questions = questions
    st.session_state.idx = 0
    st.session_state.results = []
    st.session_state.started_at = datetime.now().isoformat(timespec="seconds")
    st.session_state.stage = "interview"
    st.rerun()


def screen_interview():
    questions = st.session_state.questions
    idx = st.session_state.idx
    total = len(questions)
    q = questions[idx]
    already_scored = idx < len(st.session_state.results)

    st.progress(idx / total, text=f"Question {idx + 1} of {total}")
    st.caption(f"Topic: {q['topic']}")
    st.subheader(q["question"])

    if not already_scored:
        answer = st.text_area(
            "Your answer",
            key=f"answer_{idx}",
            height=180,
            placeholder="Type your answer here...",
        )

        if st.button("Submit answer", type="primary", disabled=not answer.strip()):
            with st.spinner("Evaluating your answer locally (this can take a little while)..."):
                try:
                    result = evaluate_answer(
                        question=q["question"],
                        reference_answer=q["reference_answer"],
                        candidate_answer=answer,
                    )
                except ValueError as err:
                    st.error(f"{err}")
                    return
                except Exception as err:  # Ollama down, model missing, guard block...
                    st.error(
                        f"Could not reach the local model: {err}\n\n"
                        f"Check that Ollama is running (`ollama serve`) and the model is "
                        f"installed (`ollama pull {MODEL_NAME}`)."
                    )
                    return

            st.session_state.results.append(
                {**result, "reference_answer": q["reference_answer"], "topic": q["topic"]}
            )
            st.rerun()
    else:
        result = st.session_state.results[idx]
        st.text_area("Your answer", value=result["candidate_answer"], height=120, disabled=True)
        render_feedback(result)

        is_last = idx == total - 1
        label = "Finish interview" if is_last else "Next question ➡️"
        if st.button(label, type="primary"):
            if is_last:
                st.session_state.stage = "report"
            else:
                st.session_state.idx += 1
            st.rerun()


def screen_report():
    results = st.session_state.results
    name = st.session_state.candidate_name or "Candidate"
    avg = round(sum(r["final_score"] for r in results) / len(results), 1)

    st.title("📋 Interview Report")
    st.write(f"**{name}** · {st.session_state.started_at}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall score", f"{avg}/10")
    c2.metric("Verdict", verdict(avg))
    c3.metric("Questions", len(results))

    df = pd.DataFrame(
        [
            {
                "Topic": r["topic"],
                "Question": r["question"],
                "Status": STATUS_LABEL.get(r["answer_status"], r["answer_status"]),
                "Correctness": r["correctness_score"],
                "Depth": r["knowledge_depth_score"],
                "Final": r["final_score"],
                "Confidence": r["confidence"],
            }
            for r in results
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    low_conf = [r for r in results if r["confidence"] == "low"]
    if low_conf:
        st.warning(f"{len(low_conf)} answer(s) were scored with low confidence; consider reviewing them manually.")

    st.subheader("Question-by-question feedback")
    for i, r in enumerate(results, 1):
        with st.expander(f"Q{i}. {r['question']}  ·  {r['final_score']}/10"):
            st.markdown(f"**Your answer:** {r['candidate_answer']}")
            st.markdown(f"**Reference answer:** {r['reference_answer']}")
            render_feedback(r)

    report = {
        "candidate": name,
        "started_at": st.session_state.started_at,
        "model": MODEL_NAME,
        "overall_score": avg,
        "results": results,
    }
    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Download JSON",
        json.dumps(report, indent=2, ensure_ascii=False),
        file_name="interview_report.json",
        mime="application/json",
        use_container_width=True,
    )
    d2.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False),
        file_name="interview_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if st.button("Start a new interview", use_container_width=True):
        reset_interview()
        st.rerun()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
init_state()

with st.sidebar:
    st.header("Settings")
    st.write(f"**Model:** `{MODEL_NAME}`")
    st.caption("Change it with the LOCAL_MODEL environment variable before launching.")
    st.divider()
    if st.session_state.stage != "setup" and st.button("Abandon / restart"):
        reset_interview()
        st.rerun()

{"setup": screen_setup, "interview": screen_interview, "report": screen_report}[
    st.session_state.stage
]()