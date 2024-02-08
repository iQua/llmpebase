"""
Implementation of the text generation with Llama2 model under the HuggingFace.
See https://huggingface.co/blog/llama2 for details.

https://huggingface.co/blog/llama2#how-to-prompt-llama-2
"""

from llmpebase.model.LM import llama2_meta

from transformers import AutoTokenizer, AutoModelForCausalLM


# To be implement
class llama2Request(llama2_meta.LLaMARequest):
    """A class to make request on the LLaMA-V2 model."""

    def __init__(self, model_config: dict) -> None:
        super().__init__(model_config)

        model_type = model_config["model_type"]
        model_name = model_config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"{model_type}/{model_name}", use_auth_token=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            f"{model_type}/{model_name}", use_auth_token=True, device_map="auto"
        )
