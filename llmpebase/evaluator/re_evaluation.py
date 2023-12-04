"""
A base evaluator using `re` of Python of achieve the evaluation.
"""

from llmpebase.evaluator import base


def format_number(number: str):
    """Convert string to be the desired mathematical format."""
    if "." in number:
        return float(number)

    return int(number)


class GSM8KEvaluator(base.BaseEvaluator):
    """A base evaluator for the GSM8K dataset."""

    def measure(self, result, groundtruth):
        return format_number(result) == format_number(groundtruth)


class GSM8KLLMEvaluator(base.BaseLLMEvaluator):
    """A evaluator implemented by LLMs for the GSM8K dataset."""

    def measure(self, result, groundtruth):
        # TO BE IMPLEMENTED
        pass
