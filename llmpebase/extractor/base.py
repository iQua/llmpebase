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

    extract_target: str = "pure mathematical types, such as an integer, float, fractional expression or a mathematical expression"

    system_prompt = "You are a powerful AI extractor in math responsible for identifying and extracting the core solution for a question from a long text answer. Please summarize the answer and extract the final solution as {}. Please only maintain the original content without making any modifications. One important hint is that the final solution generally appears at the end of the answer."

    polish_system_prompt = "You are a powerful string polisher in math responsible for processing a string into mathematical format. Please polish the given string into {}. Please remove useless, invalid, or additional strings or specific characters such as $. Just output the given string once there is no need to polish."

    head: str = "This is the problem of {}."

    extraction_head: str = "Extracted solution: "
    notice: str = "Directly return the extracted solution without any modifications."

    polish_answer: str = "The polished string is:"
    polish_instruction: str = "Polish this string into {}, without including any additional characters and words, such as $. Please recall the following special cases: \n1). When the string is a description, please convert it to value. For example, for 'There are three balls', you should output 3. \n2). When the string is an equation such as n=5 or n=\\frac{{1}}{{6}}, the polished outcomes should be 5, 1/6, respectively.\n3). For a string such as $\\frac{{3}}{{10}}$, the polished outcomes should be 3/10."
    polish_notice: str = "Return a single string as the polished result. Return the given string directly if there is no need to polish."

    def __init__(self, llm_model):
        # Define the request model used as the extractor
        self.llm_model = llm_model

    def organize_prompt(self, answer: str, **kwargs):
        """Organize the prompt for the LLM to extract the solution."""
        problem_name = kwargs["problem_name"] if "problem_name" in kwargs else "math"
        question = kwargs["question"] if "question" in kwargs else ""
        head = self.head.format(problem_name)
        prompt = f"""{head} Extract the final solution from the answer of the question.\n Notice: {self.notice}\nThe question and answer pair is as follows:\n\nQuestion: {question}\nAnswer: {answer}\n\n.{self.extraction_head}"""

        return prompt

    def organize_polish_prompt(self, solution):
        """Organize the prompt for the LLM to polish the solution."""
        polish_instruction = self.polish_instruction.format(self.extract_target)
        prompt = f"""For the string: {solution}. {polish_instruction}\nNotice: {self.polish_notice}\n{self.polish_answer}"""

        return prompt

    def forward(self, answer: str, per_request_responses: int = 1, **kwargs):
        """Performing the request."""

        # First extract the solution from the answer
        prompt = self.organize_prompt(answer, **kwargs)

        system_prompt = self.system_prompt.format(self.extract_target)
        responses = self.llm_model.forward(
            user_prompt=prompt,
            per_request_responses=per_request_responses,
            sys_prompt=system_prompt,
        )
        extracted_solution = self.llm_model.read_response_contents(responses)[0]

        # Then polish the solution
        polish_prompt = self.organize_polish_prompt(extracted_solution)
        polish_system_prompt = self.polish_system_prompt.format(self.extract_target)
        responses = self.llm_model.forward(
            user_prompt=polish_prompt,
            per_request_responses=per_request_responses,
            sys_prompt=polish_system_prompt,
        )
        solution = self.llm_model.read_response_contents(responses)[0]

        return solution
