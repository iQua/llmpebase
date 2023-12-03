"""
An extractor relying LLMs to extract the target result from a long response.
"""


class LLMExtractor:
    """A extractor relying on LLM to extract the target results from a long response."""

    system_prompt = "You are a powerful AI extractor in math responsible for identifying and extracting the core solution for a question from a long text answer. Please extract the final solution presenting as either an integer, float, an equation, or a mathematical expression. Please only maintain the original content without making any modifications. One important rule is that the final solution generally exists in the last sentence of the answer."

    def __init__(self, request_model) -> None:
        # Define the LLM model used to perform the request
        self.request_model = request_model

    def organize_prompt(self, answer: str, **kwargs):
        """Organize the prompt for the LLM."""
        question = None if "question" not in kwargs else kwargs["question"]
        prompt = f"""Extract the final solution from the answer of the question.\n The question and answer pair is as follows:\n\nQuestion: {question}\n\nAnswer: {answer}\n\n."""

        return prompt

    def perform_extraction(self, answer: str, per_request_responses: int = 1, **kwargs):
        """Performing the request."""

        prompt = self.organize_prompt(answer, **kwargs)

        responses = self.request_model.perform_request(
            user_prompt=prompt,
            per_request_responses=per_request_responses,
            sys_prompt=self.system_prompt,
        )
        extracted_solution = self.request_model.extract_response_contents(responses)[0]

        return extracted_solution
