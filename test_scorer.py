
import pytest

from scorer import evaluate_answer


QUESTION = "What is overfitting in Machine Learning?"

REFERENCE = (
    "Overfitting occurs when a model fits training data "
    "too closely and performs poorly on unseen data."
)


@pytest.mark.parametrize(
    "candidate_answer, expected_status",
    [
        (
            "The model memorizes training data and "
            "does not generalize well to new data.",
            "complete",
        ),
        (
            "Overfitting happens when a model performs "
            "very well on its training data.",
            "partially_correct",
        ),
        (
            "Overfitting means the model always performs "
            "perfectly on new, unseen data.",
            "incorrect",
        ),
    ],
)
def test_answer_categories(candidate_answer, expected_status):
    result = evaluate_answer(
        question=QUESTION,
        reference_answer=REFERENCE,
        candidate_answer=candidate_answer,
    )

    assert result["answer_status"] == expected_status
    assert 0 <= result["correctness_score"] <= 10
    assert 0 <= result["knowledge_depth_score"] <= 10
    assert 0 <= result["final_score"] <= 10
    assert result["confidence"] in ["high", "medium", "low"]


def test_equivalent_wording():
    answer = (
        "A model learns the training examples too closely "
        "and struggles with unseen examples."
    )

    result = evaluate_answer(
        QUESTION,
        REFERENCE,
        answer,
    )

    # The intended meaning is equivalent to the reference.
    assert result["correctness_score"] >= 7


def test_empty_answer_is_rejected():
    with pytest.raises(ValueError):
        evaluate_answer(QUESTION, REFERENCE, "   ")
