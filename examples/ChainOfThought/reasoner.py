"""
A reasoner to perform the reasoning process.
"""

from llmpebase.model import define_model
from llmpebase.model.LM.base import BaseLlmRequest
from llmpebase.model.prompting.base import BasicSamplePrompt


class CoTReasoner:
    """A CoT reasoner to answer the question with the request model."""

    def __init__(self, llm_model: BaseLlmRequest = None, model_config: dict = None):
        # Either one should be defined
        assert llm_model is not None or model_config is not None

        # Define the model once no llm_model is provided
        if llm_model is None:
            llm_model = define_model(model_config)

        self.llm_model = llm_model

    def forward(self, prompt_sample: BasicSamplePrompt, **kwargs):
        """Answer the question with the CoT reasoner."""
        # Generate the request prompt
        input_message = self.llm_model.create_format_input(
            user_prompt=str(prompt_sample),
            sys_prompt="""Answer the given question and get the final solution.""",
        )

        # Do model request
        responses = self.llm_model.forward(
            input_request=input_message, per_request_responses=2
        )
        return self.llm_model.read_response_contents(responses)

    def get_cost_statistics(self, **kwargs):
        """Get the cost statistics of the model."""
        data = self.llm_model.get_cost_statistics()
        self.llm_model.reset_cost_statistics()
        return data
