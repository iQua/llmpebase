"""
Implementation of text generation with llama by using the 
pipeline.
"""

import torch
from transformers import pipeline, LlamaTokenizer

from vgbase.utils.folder_utils import directory_contains_subfolder


from llmpebase.model.LM import llama_falcon


class LLaMAPipelineRequest(llama_falcon.LLaMARequest):
    """A class to forward the LLaMA model."""

    def load_model(self, model_config: dict):
        """loading the llama models."""

        model_name = model_config["model_name"]
        checkpoint_dir = model_name
        if "pretrained_models_dir" in model_config and directory_contains_subfolder(
            model_config["pretrained_models_dir"], model_name
        ):
            checkpoint_dir = model_config["pretrained_models_dir"]

        model_type = model_config["model_type"]
        assert model_type in ["llamav2"]

        tokenizer = LlamaTokenizer.from_pretrained(
            checkpoint_dir,
            use_fast=False,
            padding_side="left",
        )
        model = pipeline(
            "text-generation",
            model=checkpoint_dir,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        return model, tokenizer

    def perform_request(
        self,
        input_request: str = None,
        user_prompt: str = None,
        per_request_responses: int = 1,
        **kwargs,
    ):
        """Forwarding the model to perform a request."""

        if input_request is None and user_prompt is None:
            raise ValueError("Either request_input or user_prompt should be provided")

        model_input = (
            self.create_format_input(user_prompt)
            if input_request is None
            else input_request
        )

        responses = self.model(
            model_input,
            num_return_sequences=per_request_responses,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.generation_config,
        )
        return responses
