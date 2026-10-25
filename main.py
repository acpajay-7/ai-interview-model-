
import json

from network_guard import enable_network_guard

enable_network_guard()

from scorer import evaluate_answer
from scorer import evaluate_answer


question = "What is overfitting in Machine Learning?"

reference_answer = (
    "Overfitting occurs when a model learns the training data "
    "too closely, including noise, and performs poorly on "
    "unseen data."
)

candidate_answer = (
    "Overfitting happens when a model memorizes training "
    "examples and fails to generalize to new data."
)


try:
    result = evaluate_answer(
        question=question,
        reference_answer=reference_answer,
        candidate_answer=candidate_answer,
    )

    print("\nINTERVIEW EVALUATION")
    print("=" * 40)
    print(json.dumps(result, indent=4, ensure_ascii=False))

except Exception as error:
    print("Evaluation failed:", error)
