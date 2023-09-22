"""
Getting the pre-trained LLaMA model for inference.

Please check https://zhuanlan.zhihu.com/p/653926703
for hyper-parameters settings.

"""

from typing import List

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
import tensor_parallel as tp

from vgbase.utils.folder_utils import directory_contains_subfolder


from llmpebase.models.LMs import base


class LLaMARequest(base.BaseLMRequest):
    """A class to forward the LLaMA model."""

    def __init__(self, model_config: dict, envs_config: dict) -> None:
        super().__init__(model_config, envs_config)

        self.model, self.tokenizer = self.load_model(model_config, envs_config)

        self.model.eval()

    def get_generation_config(self):
        """Getting the model request config."""

        generation_settings = self.model_config["generation_settings"]
        self.generation_config = generation_settings
        # set the necessary hyper-parameters
        temperature = (
            generation_settings["temperature"]
            if "temperature" in generation_settings
            else 0.7
        )
        max_new_tokens = (
            generation_settings["max_new_tokens"]
            if "max_new_tokens" in generation_settings
            else 1000
        )
        top_p = generation_settings["top_p"] if "top_p" in generation_settings else 0.75
        top_k = generation_settings["top_k"] if "top_k" in generation_settings else 40
        num_beams = (
            generation_settings["num_beams"]
            if "num_beams" in generation_settings
            else 4
        )

        # set basic default settings for gpt
        self.generation_config.update(
            {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "num_beams": num_beams,
            }
        )

    def load_model(self, model_config: dict, envs_config: dict):
        """loading the llama models."""
        model_name = model_config["model_name"]
        checkpoint_dir = model_name
        if "pretrained_models_dir" in model_config and directory_contains_subfolder(
            model_config["pretrained_models_dir"], model_name
        ):
            checkpoint_dir = model_config["pretrained_models_dir"]

        model_type = model_config["model_type"]
        assert model_type in ["llama", "falcon"]

        tokenizer = LlamaTokenizer.from_pretrained(
            checkpoint_dir,
            use_fast=False,
            padding_side="left",
        )
        tokenizer.pad_token_id = (
            0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        )
        tokenizer.bos_token_id = 1

        if model_type == "llama":
            # we use tensor parallel for loading llama
            model = LlamaForCausalLM.from_pretrained(
                checkpoint_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder="offload",
            )
            if envs_config["distributed"]:
                n_gpus = envs_config["world_size"]
                model = tp.tensor_parallel(model, [i for i in range(n_gpus)])

        else:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_dir,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

        return model, tokenizer

    def create_format_prompt(self, instruction: str, prompts: List[str]):
        """Creating prompts for the LLaMA models."""
        instruction = f"{instruction}\n\n"
        format_prompt = "\n".join(prompts)
        return format_prompt

    def get_tokens_input(self, prompts):
        """Getting the tokens of prompts"""
        input_tokens = self.tokenizer.batch_encode_plus(
            prompts, return_tensors="pt", padding=True
        )

        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                if self.envs_config["device"] == "cuda":
                    input_tokens[t] = input_tokens[t].to("cuda")
                if self.envs_config["device"] == "mps":
                    input_tokens[t] = input_tokens[t].to("mps")

        return input_tokens

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
        model_inputs = [model_input for _ in range(per_request_responses)]

        encode_inputs = self.get_tokens_input(model_inputs)

        generate_ids = self.model.generate(
            **encode_inputs,
            generation_config=GenerationConfig(**self.generation_config),
        )

        responses = self.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
        )
        print("responses: ", responses)
        return responses

    def create_format_input(self, user_prompt, **kwargs):
        """Creating the format input received by the model."""

        instruct_prompt = "Follow the given examples and answer the question."
        if "instruct_prompt" in kwargs and kwargs["instruct_prompt"] is not None:
            instruct_prompt = kwargs["sys_prompt"]

        # prompt = f"""{instruct_prompt} Please utilize a sub-sentence '{self.target_answer_format}' to point out the final solution for users to read. \n\n {user_prompt}"""
        prompt = "What is the ChapGPT?"
        return prompt

    def extract_answers(self, responses: list):
        """Extracting answer from the response of the model."""
        answers = []
        for res in responses:
            answers.extend(choice["message"]["content"] for choice in res["choices"])
        return answers
