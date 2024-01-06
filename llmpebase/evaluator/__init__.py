"""
An interface to evaluate the performance by measuring the similarity between results and 
groundtruths.
"""
import logging

from llmpebase.evaluator.re_evaluation import (
    GeneralEvaluator,
)


basic_evaluators = {
    "GSM8K": GeneralEvaluator,
    "MMLU": GeneralEvaluator,
    "MATH": GeneralEvaluator,
    "BBH": GeneralEvaluator,
    "TheoremQA": GeneralEvaluator,
    "GameOf24": GeneralEvaluator,
}

llm_evaluators = {
    "GSM8K": "Not implemented",
    "MMLU": "Not implemented",
    "MATH": "Not implemented",
    "BBH": "Not implemented",
    "TheoremQA": "Not implemented",
    "GameOf24": "Not implemented",
}


def get(data_name, style="basic", **kwargs):
    """Get the evaluators for the specific dataset."""

    assert style in ["basic", "llm"]

    evaluators = basic_evaluators if style == "basic" else llm_evaluators

    if data_name in evaluators:
        logging.info("Get %s evaluator for %s", style, data_name)
        return evaluators[data_name]

    else:
        raise NotImplementedError(
            f"{style} evaluator is not implemented for {data_name}"
        )
