"""
The implementation of base model for LMs.
"""

from typing import Union, List

import torch


class BaseLlmRequest(torch.nn.Module):
    """The basic request model for large language model."""

    def __init__(self, model_config: dict):
        super().__init__()

        # Get the model information
        self.model_config = model_config
        self.model_name = model_config["model_name"]

        # Configuration for the generation
        self.generation_config = {}

        # Basic components to record the resource used
        # Number of requests
        self.num_requests = 0
        # Number of words in the system and user's prompts
        self.num_words = {"system": [], "user": []}
        # Number of tokens in the system and user's prompts
        self.num_prompt_tokens = []
        # Number of tokens in the system and user's prompts
        self.num_completion_words = []
        self.num_completion_tokens = []

    def configuration(self):
        """Configuration of the model."""
        # 1. Temperature [0, 1] is a parameter that controls the "creativity" or randomness
        # of the text generated by GPT-3. A higher temperature (e.g., 0.7) results in
        # more diverse and creative output, while a lower temperature (e.g., 0.2) makes
        # the output more deterministic and focused.
        # In practice, temperature affects the probability distribution over the
        # possible tokens at each step of the generation process. A temperature of 0
        # would make the model completely deterministic, always choosing the most
        # likely token.
        # Low temperature (0 to 0.3): More focused, coherent, and conservative outputs.
        # Medium temperature (0.3 to 0.7): Balanced creativity and coherence.
        # High temperature (0.7 to 1): Highly creative and diverse, but potentially
        #   less coherent.

        # 2. Top P ranges from 0 to 1 (default), and a lower Top P means the model
        # samples from a narrower selection of words. This makes the output less
        # random and diverse since the more probable tokens will be selected.
        # 3. Max tokens determine the maximum length of the generated text.
        self.generation_config = {
            "temperature": 0.7,
            "max_tokens": 500,
            "top_p": 0.7,
        }

    def create_format_input(self, user_prompt, **kwargs):
        """Creating the format input received by the"""
        raise NotImplementedError("'create_format_input' has not been implemented yet.")

    def forward(
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
        raise NotImplementedError("'forward' has not been implemented yet.")

    def read_response_contents(self, responses: list):
        """Read main contents from the obtained responses."""
        raise NotImplementedError("'extract_answers' has not been implemented yet.")

    def compute_costs(self, input_messages: dict, responses: list):
        """Compute the costs made by the requests."""
        raise NotImplementedError("'extract_tokens' has not been implemented yet.")

    def is_limit_request(self):
        """Whether the request model has limited request rate."""

        return False

    def get_latest_cost(self):
        """Get the latest cost statistics."""
        return {
            "num_requests": 1,
            "num_words": {key: self.num_words[key][-1] for key in self.num_words},
            "num_prompt_tokens": self.num_prompt_tokens[-1],
            "num_completion_words": self.num_completion_words[-1],
            "num_completion_tokens": self.num_completion_tokens[-1],
        }

    def get_cost_statistics(self, latest=False):
        """Save the statistics of the requests into the json file."""
        if latest:
            return self.get_latest_cost()

        return {
            "num_requests": self.num_requests,
            "num_words": self.num_words,
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_completion_words": self.num_completion_words,
            "num_completion_tokens": self.num_completion_tokens,
        }

    def reset_cost_statistics(self):
        """Reset the cost statistics."""
        self.num_requests = 0
        self.num_words = {"system": [], "user": []}
        self.num_prompt_tokens = []
        self.num_completion_words = []
        self.num_completion_tokens = []
