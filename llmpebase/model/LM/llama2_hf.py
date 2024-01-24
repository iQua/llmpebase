"""
Implementation of the text generation with Llama2 model under the HuggingFace.
See https://huggingface.co/blog/llama2 for details.
"""

from llmpebase.model.LM import llama2_meta

from transformers import AutoTokenizer, AutoModelForCausalLM


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

    def create_format_input(self, user_prompt: str, **kwargs):
        """Creating messages to be used for forwarding."""

        sys_prompt = "Follow the given prompt to generate correct response."
        sys_prompt = f"""{sys_prompt}. Please utilize a sub-sentence '{self.solution_flag}' to point out the core response for users to read. """

        if "sys_prompt" in kwargs and kwargs["sys_prompt"] is not None:
            sys_prompt = kwargs["sys_prompt"]

        sys_message = {
            "role": "system",
            "content": sys_prompt,
        }
        requeset_message = {"role": "user", "content": user_prompt}

        request_messages = [
            sys_message,
            requeset_message,
        ]

        return request_messages

    def forward(
        self,
        input_request: str = None,
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

        print("-------")
        print("input_dialogs: ", input_dialogs)
        responses = self.model.chat_completion(
            input_dialogs,
            **self.generation_config,
        )
        return responses

    def extract_response_contents(self, responses: list):
        """Extracting answer from the response of the model."""
        print("-------- raw responses: ")
        print(responses)
        print("---------------------")

        contents = []
        for res in responses:
            contents.append(res["generation"]["content"])
        return contents

    def extract_tokens(self, responses: list):
        """Extracting tokens from the responses."""
        return None
