"""
Base extractors to be inherited by the specific extractor.
"""

from typing import Tuple, Any

from llmpebase.model.LM.base import BaseLlmRequest


class BaseReExtractor:
    """The base extractor built upon the `re` of Python to extract the groundtruth from the raw answer."""

    def forward(self, answer: Any, **kwargs) -> Tuple[str, str, str]:
        """Extract the groundtruth from the raw answer.

        :return answer, conclusion, groundtruth in which
         answer is the direct answer to the question containing all contents
         conclusion is the summary or final step of the answer
         groundtruth is the solution obtained by the answer
        """
        raise NotImplementedError("An implementation of the extractor is required.")


class BaseLlmExtractor:
    """The base extractor built upon the LLM to extract the target result from the response."""

    def __init__(self, llm_model: BaseLlmRequest):
        # Define the request model used as the extractor
        self.llm_model = llm_model

    def forward(self, answer, per_request_responses: int = 1, **kwargs):
        """Extract the target result from the response."""
        raise NotImplementedError("An implementation of the extractor is required.")
