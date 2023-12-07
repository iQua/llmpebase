"""
A reasoner to perform the reasoning process.
"""

from llmpebase.model import define_model
from llmpebase.model.LM.base import BaseLlmRequest


class CoTReasoner:
    """A CoT reasoner to answer the question with the request model."""

    def __init__(self, llm_model: BaseLlmRequest = None, model_config: dict = None):
        # Either one should be defined
        assert llm_model is not None or model_config is not None

        # Define the model once no llm_model is provided
        if llm_model is None:
            llm_model = define_model(model_config)

        self.llm_model = llm_model

    def forward(self, request_prompt: str):
        """Answer the question with the CoT reasoner."""
        # Generate the request prompt
        input_message = self.llm_model.create_format_input(
            user_prompt=request_prompt,
            sys_prompt="""Answer the given question and get the final solution.""",
        )

        # Do model request
        responses = self.llm_model.forward(
            input_request=input_message, per_request_responses=2
        )
        return self.llm_model.read_response_contents(responses)
