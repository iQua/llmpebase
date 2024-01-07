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


class CSQARespLlmExtractor(base.BaseLlmExtractor):
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


class GameOf24RespLlmExtractor(base.BaseLlmExtractor):
    """A extractor relying on LLM to extract the target results from a long response for the GameOf24 dataset."""

    extract_target: str = "a pure mathematical equation containing brackets"

    system_prompt = "You are a powerful AI summarizer in the Game of 24 responsible for summarizing and formatting the reasoning steps into the final solution presented as {}. Please only summarize based on the original content without introducing any modifications. One important hint is that the brackets on the final equation depend on the order of reasoning steps."

    head: str = "This is the problem of {}."

    extraction_head: str = "Extracted equation: "
    notice: str = "Directly output the summarized equation without any modifications."

    polish_answer: str = "The polished string is:"
    polish_instruction: str = "Polish this string into {}, without including any additional characters and words, such as $. Please only maintain the necessary brackets. "
    polish_notice: str = "Return a single string as the polished result. Return the given string directly if there is no need to polish."
