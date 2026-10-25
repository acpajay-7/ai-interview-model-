
import json
import logging
import os
from typing import Literal

import ollama
from pydantic import BaseModel, Field, ValidationError

# Configuration

MODEL_NAME = os.getenv("LOCAL_MODEL", "deepseek-r1:8b")

logging.basicConfig(
    filename= "scorer.log",
    level= logging.INFO,
    format= "%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)

# Defining the expected output structure

class Evaluation(BaseModel):
    question: str
    candidate_answer: str

    correctness_score: float = Field(ge=0, le=10)
    knowledge_depth_score: float = Field(ge=0, le=10)

    answer_status: Literal[
        "complete",
        "partially_correct",
        "incorrect",
        "uncertain",
    ]

    explanation: str
    unsupported_claims: list[str]
    contradictions: list[str]

    confidence: Literal["high", "medium", "low"]
    confidence_reason: str

    final_score: float = Field(ge=0, le=10)


# 3. Main scoring function

def evaluate_answer(
    question: str,
    reference_answer: str,
    candidate_answer: str,
) -> dict:

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not reference_answer.strip():
        raise ValueError("Reference answer cannot be empty.")

    if not candidate_answer.strip():
        raise ValueError("Candidate answer cannot be empty.")

    prompt = f"""
You are a careful technical interview evaluator.

Evaluate the candidate's answer using the question,
reference answer, and scoring rubric below.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

CANDIDATE ANSWER:
{candidate_answer}

RUBRIC:

1. Correctness: score from 0 to 10.
   9-10: Correct and accurate.
   7-8: Mostly correct, with minor omissions.
   4-6: Partially correct or missing important concepts.
   1-3: Mostly incorrect, with little correct understanding.
   0: Completely incorrect or irrelevant.

2. Knowledge and depth: score from 0 to 10.
   Assess relevant understanding, explanation,
   reasoning, useful detail, and examples when relevant.

3. Accept equivalent wording, valid alternative approaches,
   and correct examples. Do not require exact keywords.

4. Do not award marks simply because the candidate repeats
   words from the reference answer.

5. Identify unsupported factual claims and contradictions.
   Do not label a claim unsupported merely because it is
   absent from the reference answer. Assess it carefully.

6. The reference answer is a guide, not necessarily the
   only valid answer. Do not invent sources or claim to
   have searched the internet.

7. If the question or reference answer is insufficient to
   judge a claim reliably, lower confidence and explain why.

8. "complete" means the answer covers the key expected
   concepts. "partially_correct" means some key concepts
   are correct but others are missing or mistaken.
   "incorrect" means the answer is substantially wrong
   or irrelevant. Use "uncertain" when reliable evaluation
   is not possible.

9. Calculate final_score as:
   correctness_score * 0.7 + knowledge_depth_score * 0.3

10. Round scores to one decimal place.

Return ONLY a JSON object matching this structure:
{{
  "question": "original question",
  "candidate_answer": "original candidate answer",
  "correctness_score": 0.0,
  "knowledge_depth_score": 0.0,
  "answer_status": "complete",
  "explanation": "clear reasons for the scores",
  "unsupported_claims": [],
  "contradictions": [],
  "confidence": "medium",
  "confidence_reason": "reason for confidence level",
  "final_score": 0.0
}}
"""

    #  Ollama on the local machine.
    # It is NOT an external AI API request.
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You evaluate interview answers carefully. "
                    "Follow the rubric and return valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        format="json",
        options={"temperature": 0},
    )

    raw_output = response["message"]["content"]

    try:
        data = json.loads(raw_output)
        evaluation = Evaluation.model_validate(data)

    except (json.JSONDecodeError, ValidationError) as exc:
        logger.error("Invalid model output: %s", exc)
        raise ValueError(
            "The local model returned an invalid evaluation. "
            "Please retry or inspect the model output."
        ) from exc

   
    calculated_score = round(
        evaluation.correctness_score * 0.7
        + evaluation.knowledge_depth_score * 0.3,
        1,
    )

    result = evaluation.model_dump()
    result["final_score"] = calculated_score

    logger.info(
        "Local evaluation completed; confidence=%s",
        result["confidence"],
    )

    return result
