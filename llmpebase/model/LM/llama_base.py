"""
Base implementation of the Llama model.
"""

from llmpebase.model.LM import base


class LlamaRequest(base.BaseLlmRequest):
    """A class to forward the LLaMA model."""

    def configuration(self):
        """Configure the Llama model."""
        # Get the basic configuration
        super().configuration()

        generation_settings = self.model_config["generation_settings"]

        # Change the
        # max_seq_len: the maximum length of input sequences.
        #   All models support sequence length up to 4096 tokens, but we pre-allocate
        #   the cache according to max_seq_len and max_batch_size values. So set those
        #   according to your hardware
        # max_gen_len: the maximum length of generated sequences
        # top_k: This is the number of probable next words, to create a pool of words
        # to choose from, default to be 40
        generation_settings["max_gen_len"] = generation_settings.pop("max_tokens")

        self.generation_config.update(generation_settings)

    def create_format_input(self, user_prompt, **kwargs):
        """Creating the format input received by the"""
        raise NotImplementedError("'create_format_input' has not been implemented yet.")

    def read_response_contents(self, responses: list):
        """Read main contents from the obtained responses."""
        raise NotImplementedError("'extract_answers' has not been implemented yet.")

    def compute_costs(self, input_messages: dict, responses: list):
        """Count answers from the obtained responses."""
        raise NotImplementedError("'extract_tokens' has not been implemented yet.")

    def is_limit_request(self):
        """Whether the request model has limited request rate."""

        return False
