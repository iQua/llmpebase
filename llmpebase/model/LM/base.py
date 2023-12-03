"""
The implementation of base model for LMs.
"""
import re
from typing import Union, List


class BaseLMRequest:
    """The basic request model for large language model."""

    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.model_name = model_config["model_name"]

        # a pre-defined format for the request answer
        # this is mainly used to extract the target answer from the
        # response easier
        self.target_answer_format = ""

        self.generation_config = {}
        self.get_generation_config()

    def get_generation_config(
        self,
    ):
        """Setting the request config for the model."""

    def set_target_answer_format(self, solution_format: str = "The answer is: ."):
        """Setting the target answer format."""
        self.target_answer_format = solution_format

    def perform_request(
        self,
        input_request: Union[List[dict], str] = None,
        user_prompt: str = None,
        per_request_responses: int = 1,
        **kwargs,
    ):
        """Performing request once.

        :param request_input: The input of the model to perform a request. As this is the
         directly input, the type of this argument should depends on which model is used
         to perform the request.
         For ChatGPTs, the request_input should be the `List[dict]` containing multiple terms
         For Llama, the request_input should be the a string

        :param user_prompt: A `string` containing the prompt defined by the user, otherwise,
         it maybe not be the desired input of the model to perform a request. Thus, further
         processing should be implemented when necessary.

        :param per_request_responses: A `int` showing how many responses will be returned by
         the model.
        """
        raise NotImplementedError("'perform_request' has not been implemented yet.")

    def create_format_input(self, user_prompt, **kwargs):
        """Creating the format input received by the"""
        raise NotImplementedError("'create_format_input' has not been implemented yet.")

    def extract_response_contents(self, responses: list):
        """Extracting main contents from the obtained responses."""
        raise NotImplementedError("'extract_answers' has not been implemented yet.")

    def extract_tokens(self, responses: list):
        """Extracting answers from the obtained responses."""
        raise NotImplementedError("'extract_tokens' has not been implemented yet.")

    def has_request_limit(self):
        """Whether the request model has limited request rate."""

        return False

    def extract_target_answers(self, responses_content: list):
        """Extracting the target answer from the contents of responses."""
        prefix = re.escape("In summary")
        # 1. extract the string after the answer format
        pattern = rf"{prefix}\s*(.+)"

        obtained_targets = []
        for content in responses_content:
            match = re.search(pattern, content, re.IGNORECASE)

            obtained_targets.append(match.group(1) if match else content)

        return obtained_targets
