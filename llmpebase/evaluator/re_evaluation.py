"""
A base evaluator using `re` of Python of achieve the evaluation.
"""

from llmpebase.evaluator import base


def format_number(number: str):
    """Convert string to be the desired mathematical format."""
    # Remove trailing dot (if any)
    # To address the condition like "16.00."
    if number.endswith("."):
        number = number.rstrip(".")
    if "." in number:
        return float(number)

    return int(number)


class GSM8KEvaluator(base.BaseEvaluator):
    """A base evaluator for the GSM8K dataset."""

    def measure(self, result, groundtruth):
        return format_number(result) == format_number(groundtruth)


class GSM8KLlmEvaluator(base.BaseLLMEvaluator):
    """A evaluator implemented by LLMs for the GSM8K dataset."""

    def measure(self, result, groundtruth):
        # TO BE IMPLEMENTED
        pass


class MMLUEvaluator(base.BaseEvaluator):
    """A base evaluator for the MMLU dataset."""

    def measure(self, result, groundtruth):
        return result == groundtruth


class MMLULlmEvaluator(base.BaseLLMEvaluator):
    """A evaluator implemented by LLMs for the GSM8K dataset."""

    def measure(self, result, groundtruth):
        # TO BE IMPLEMENTED
        pass


class MATHEvaluator(base.BaseEvaluator):
    """A base evaluator for the MMLU dataset."""

    def measure(self, result, groundtruth):
        return result == groundtruth


class MATHLlmEvaluator(base.BaseLLMEvaluator):
    """A evaluator implemented by LLMs for the GSM8K dataset."""

    def measure(self, result, groundtruth):
        # TO BE IMPLEMENTED
        pass


class BBHEvaluator(base.BaseEvaluator):
    """A base evaluator for the MMLU dataset."""

    def measure(self, result, groundtruth):
        return result == groundtruth


class BBHLlmEvaluator(base.BaseLLMEvaluator):
    """A evaluator implemented by LLMs for the GSM8K dataset."""

    def measure(self, result, groundtruth):
        # TO BE IMPLEMENTED
        pass
