"""
A base reasoner to perform the reasoning with LLM directly.
Specifically, with the prompt as the input, the LLM will generate 
the response containing the reasoning process.
"""

from llmpebase.model import define_model
from llmpebase.model.LM.base import BaseLlmRequest
from llmpebase.model.prompting.base import BasicSamplePrompt


class BaseLLMReasoner:
    """A base reasoner to answer the question with the llm model."""

    def __init__(self, llm_model: BaseLlmRequest = None, model_config: dict = None):
        # Either one should be defined
        assert llm_model is not None or model_config is not None

        # Define the model once no llm_model is provided
        if llm_model is None:
            llm_model = define_model(model_config)

        # How many reasoning to be performed
        # Here the reasoning is the number of responses to be generated
        # by the llm as each response contains a whole reasoning process
        self.num_reasoning = (
            1 if "num_reasoning" not in model_config else model_config["num_reasoning"]
        )

        # The basic llm model to perform reasoning
        self.llm_model = llm_model

    def forward(self, prompt_sample: BasicSamplePrompt, **kwargs):
        """Answer the question with the llm."""
        # Generate the request prompt
        input_message = self.llm_model.create_format_input(
            user_prompt=str(prompt_sample),
            sys_prompt="""Answer the given question and get the final solution.""",
        )

        # Do model request
        responses = self.llm_model.forward(
            input_request=input_message, per_request_responses=self.num_reasoning
        )
        return self.llm_model.read_response_contents(responses)

    def get_cost_statistics(self, **kwargs):
        """Get the cost statistics of the model."""
        data = self.llm_model.get_cost_statistics()
        self.llm_model.reset_cost_statistics()
        return data
