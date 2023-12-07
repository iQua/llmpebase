"""
An interface to evaluate the performance by measuring the similarity between results and 
groundtruths.
"""
import logging

from llmpebase.evaluator.re_evaluation import (
    GSM8KEvaluator,
    GSM8KLlmEvaluator,
    MMLUEvaluator,
    MMLULlmEvaluator,
)


basic_evaluators = {"GSM8K": GSM8KEvaluator, "MMLU": MMLUEvaluator}

llm_evaluators = {"GSM8K": GSM8KLlmEvaluator, "MMLU": MMLULlmEvaluator}


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
