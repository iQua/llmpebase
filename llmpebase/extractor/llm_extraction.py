"""
An extractor relying LLMs to extract the target result from a long response.
"""

from llmpebase.extractor import base


class GSM8KRespLlmExtractor(base.BaseLlmExtractor):
    """A extractor relying on LLM to extract the target results from a long response."""

    notice: str = "Return the extracted solution in the mathematical format, such as integer, float, or equation."
