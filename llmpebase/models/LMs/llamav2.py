"""
Implementation of llama v2, which requires a new way to generate the text.
"""


from llama import Llama, Dialog

from llmpebase.models.LMs import llama_falcon


class LLaMAV2Request(llama_falcon.LLaMARequest):
    """A class to make request on the LLaMA-V2 model."""

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
        max_gen_len = (
            generation_settings["max_gen_len"]
            if "max_gen_len" in generation_settings
            else 512
        )
        top_p = generation_settings["top_p"] if "top_p" in generation_settings else 0.75

        # set basic default settings for gpt
        self.generation_config.update(
            {
                "temperature": temperature,
                "max_gen_len": max_gen_len,
                "top_p": top_p,
            }
        )

    def load_model(self, model_config: dict, envs_config: dict):
        """loading the llama models."""

        model_name = model_config["model_name"]

        model_dir = model_config["downloaded_model_dir"]
        tokenizer_path = model_config["downloaded_tokenizer_path"]

        generator = Llama.build(
            ckpt_dir=model_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=512,
            max_batch_size=8,
        )

        return generator, None

    def create_format_input(self, user_prompt: str, **kwargs) -> Dialog:
        """Creating messages to be used for forwarding."""

        sys_prompt = "Follow the given examples and answer the question."
        sys_prompt = f"""{sys_prompt}. Please utilize a sub-sentence '{self.target_answer_format}' to point out the final solution for users to read. """

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

        dialog: Dialog = (
            self.create_format_input(user_prompt, **kwargs)
            if input_request is None
            else input_request
        )
        input_dialogs = [dialog for _ in range(per_request_responses)]

        responses = self.model.chat_completion(
            input_dialogs,
            **self.generation_config,
        )
        return responses

    def extract_responses_content(self, responses: list):
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
