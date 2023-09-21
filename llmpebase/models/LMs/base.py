"""
The implementation of base model for LMs.
"""

from typing import Union, List


class BaseLMRequest:
    """The basic request model for large language model."""

    def __init__(self, model_config: dict, envs_config: dict):
        self.envs_config = envs_config
        self.model_name = model_config["model_name"]

        # a pre-defined format for the request answer
        # this is mainly used to extract the target answer from the
        # response easier
        self.target_answer_format = ""

    def set_target_answer_format(self, solution_format: str = "The answer is: ."):
        """Setting the target answer format."""
        self.target_answer_format = solution_format

    def perform_request(
        self,
        request_input: Union[List[dict], str] = None,
        user_prompt: str = None,
        per_request_responses: int = 1,
        **kwargs
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

    def extract_answers(self, responses: list):
        """Extracting answers from the obtained responses."""
        raise NotImplementedError("'extract_answers' has not been implemented yet.")

    def extract_tokens(self, responses: list):
        """Extracting answers from the obtained responses."""
        raise NotImplementedError("'extract_tokens' has not been implemented yet.")

    def extract_response_target_answer(self, responses: list):
        """Extracting the target answer from the responses."""
        raise NotImplementedError(
            "'extract_response_target_answer' has not been implemented yet."
        )
