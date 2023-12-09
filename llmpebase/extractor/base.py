"""
Base extractors to be inherited by the specific extractor.
"""

from typing import Tuple, Any

# from llmpebase.model.LM.base import BaseLlmRequest


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

    system_prompt = "You are a powerful AI extractor in math responsible for identifying and extracting the core solution for a question from a long text answer. Please extract the final solution presenting as either an integer, float, an equation, or a mathematical expression. Please only maintain the original content without making any modifications. One important rule is that the final solution generally exists in the last sentence of the answer."

    head: str = "This is the problem of {}."

    extraction_head: str = "Extracted solution: "

    notice: str = "Directly return the extracted solution without any modifications."

    def __init__(self, llm_model):
        # Define the request model used as the extractor
        self.llm_model = llm_model

    def organize_prompt(self, answer: str, **kwargs):
        """Organize the prompt for the LLM."""
        problem_name = kwargs["problem_name"] if "problem_name" in kwargs else "math"
        question = None if "question" not in kwargs else kwargs["question"]
        head = self.head.format(problem_name)
        prompt = f"""{head}Extract the final solution from the answer of the question.\n Notice:{self.notice}\nThe question and answer pair is as follows:\n\nQuestion: {question}\nAnswer: {answer}\n\n.{self.extraction_head}"""

        return prompt

    def forward(self, answer: str, per_request_responses: int = 1, **kwargs):
        """Performing the request."""

        prompt = self.organize_prompt(answer, **kwargs)

        responses = self.llm_model.forward(
            user_prompt=prompt,
            per_request_responses=per_request_responses,
            sys_prompt=self.system_prompt,
        )
        extracted_solution = self.llm_model.read_response_contents(responses)[0]

        return extracted_solution
