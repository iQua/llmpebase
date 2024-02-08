"""
Implementation of Llama2 based on the MetaAI's approach.
See https://github.com/facebookresearch/llama for details.
"""
from typing import List

from llama import Llama, Dialog


from llmpebase.model.LM import llama_base


class LLaMARequest(llama_base.LlamaRequest):
    """A class to forward the LLaMA model."""

    def __init__(self, model_config: dict) -> None:
        super().__init__(model_config)

        model_dir = model_config["downloaded_model_dir"]
        tokenizer_path = model_config["downloaded_tokenizer_path"]

        self.generator = Llama.build(
            ckpt_dir=model_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=1024,
            max_batch_size=8,
        )

    def forward(
        self,
        input_request: List[dict] = None,
        user_prompt: str = None,
        per_request_responses: int = 1,
        **kwargs,
    ):
        """Forwarding the model to perform a request."""

        if input_request is None and user_prompt is None:
            raise ValueError("Either request_input or user_prompt should be provided")

        dialog: Dialog = (
            self.create_format_input(user_prompt, **kwargs)
            if input_request is None
            else input_request
        )
        input_dialogs = [dialog for _ in range(per_request_responses)]

        responses = self.generator.chat_completion(
            input_dialogs,
            **self.generation_config,
        )

        # Compute the resources
        self.compute_costs(input_messages=input_dialogs, responses=responses)

        return responses

    def create_format_input(self, user_prompt, **kwargs) -> Dialog:
        """Create the format input received by the model."""
        system_prompt = "Follow the given prompt to generate correct response."

        if "sys_prompt" in kwargs and kwargs["sys_prompt"] is not None:
            system_prompt = kwargs["sys_prompt"]

        system_message = {
            "role": "system",
            "content": system_prompt,
        }
        message = {"role": "user", "content": user_prompt}
        request_messages = [
            system_message,
            message,
        ]

        return request_messages

    def read_response_contents(self, responses: list):
        """Extracting main contents from the obtained responses."""
        contents = []
        for res in responses:
            contents.extend(res["generation"]["content"])
        return contents

    def compute_costs(self, input_messages: dict, responses: list):
        """Count costs made by the requests."""
        self.num_words["system"].append(len(input_messages[0]["content"].split()))
        self.num_words["user"].append(len(input_messages[1]["content"].split()))
        completion_tokens = 0
        completion_words = 0
        prompt_tokens = 0

        for res in responses:
            prompt_tokens += res.usage.prompt_tokens
            completion_tokens += res.usage.completion_tokens
            completion_words += sum(
                [len(choice.message.content.split()) for choice in res.choices]
            )
        # Record the statistics
        self.num_prompt_tokens.append(prompt_tokens)
        self.num_completion_tokens.append(completion_tokens)
        self.num_completion_words.append(completion_words)

    def is_limit_request(self):
        """Llamas does not have the upper bound of the request rate as
        the model is open source and downloaded locally."""
        return False
