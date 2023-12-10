"""
An extractor relying LLMs to extract the target result from a long response.
"""

from llmpebase.extractor import base

from llmpebase.extractor.re_extraction import extract_flagged_conclusion


class GSM8KRespLlmExtractor(base.BaseLlmExtractor):
    """A extractor relying on LLM to extract the target results from a long response."""


class MMLURespLlmExtractor(base.BaseLlmExtractor):
    """A extractor relying on LLM to extract the target results from a long response."""

    extract_target: str = "pure options/choices, such as A-Z"


class MATHGtLlmExtractor(base.BaseLlmExtractor):
    """A extractor relying on LLM to extract the target results from a long response for the MATH dataset."""

    def forward(self, answer: str, per_request_responses: int = 1, **kwargs):
        """Performing the request."""
        groundtruth = super().forward(answer, per_request_responses, **kwargs)
        conclusion = extract_flagged_conclusion(
            answer, flags=["=", "\\boxed"], weights=[1, 3]
        )
        return answer, conclusion, groundtruth


class MATHRespLlmExtractor(base.BaseLlmExtractor):
    """A extractor relying on LLM to extract the target results from a long response for the MATH dataset."""


class BBHRespLlmExtractor(base.BaseLlmExtractor):
    """A extractor relying on LLM to extract the target results from a long response for the BBH dataset."""


class TheoremQARespLlmExtractor(base.BaseLlmExtractor):
    """A extractor relying on LLM to extract the target results from a long response for the TheoremQA dataset."""
