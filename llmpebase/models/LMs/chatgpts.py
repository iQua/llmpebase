"""
Getting the ChatGPTs API from the OPENAI.

Official instruction: 
[1]. https://platform.openai.com/docs/guides/gpt

For more details for APIs, please access:
[2]. https://platform.openai.com/docs/api-reference/introduction

From keys creation to running, a detailed instruction is presented in 
[3]. https://wandb.ai/onlineinference/gpt-python/reports/Setting-Up-GPT-4-In-Python-Using-the-OpenAI-API--VmlldzozODI1MjY4

A more brief instruction: 
[4]. https://www.datacamp.com/tutorial/using-gpt-models-via-the-openai-api-in-python
"""

import re
import logging
from typing import List

import openai

from llmpebase.models.LMs import base


class ChatGPTAPIRequest(base.BaseLMRequest):
    """A class to forward the ChatGPT model with API of OPENAI."""

    def __init__(self, model_config: dict, envs_config: dict) -> None:
        super().__init__(model_config, envs_config)

        assert self.model_name in ["gpt-3.5-turbo", "gpt-4"]

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
        max_tokens = (
            generation_settings["max_tokens"]
            if "max_tokens" in generation_settings
            else 1000
        )
        n_completions_per_prompt = (
            generation_settings["n"] if "n" in generation_settings else 1
        )
        stop = generation_settings["stop"] if "stop" in generation_settings else None

        # set basic default settings for gpt
        self.generation_config.update(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "n": n_completions_per_prompt,
                "stop": stop,
            }
        )

    def get_authorization(self, organization: str, api_key: str):
        """Getting the authorization from openai."""
        openai.organization = organization
        openai.api_key = api_key
        logging.info("Connected to OPENAI ChapGPT APIs.")
        logging.info("With organization %s.", organization)
        logging.info("With API key %s.", api_key)

    def completion_with_backoff(self, **kwargs):
        """Completing the forward within the OPENAI."""
        return openai.ChatCompletion.create(**kwargs)

    def create_format_input(self, user_prompt: str, **kwargs):
        """Creating messages to be used for forwarding.

        By default, the messages are created to be in
        the in-context learning version, i.e., given
        some examples to make the chatgpt model make predictions
        in the similar way.

        See [4] for how to organize messages.
        """
        sys_prompt = "Follow the given examples and answer the question."
        if "sys_prompt" in kwargs and kwargs["sys_prompt"] is not None:
            sys_prompt = kwargs["sys_prompt"]

        sys_prompt = f"""{sys_prompt} Please utilize a sub-sentence '{self.target_answer_format}' to point out the final solution for users to read."""

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
        input_request: List[dict] = None,
        user_prompt: str = None,
        per_request_responses: int = 1,
        **kwargs,
    ):
        """Performing one request for `per_request_responses`.

        :return model_responses: A `List` in which each item is a
         OpenAIObject, mainly containing 'choices' and 'usage'.
         The 'choices' includes all responses of the item.
        """

        if input_request is None and user_prompt is None:
            raise ValueError("Either input_request or user_prompt should be provided")

        input_messages = (
            self.create_format_input(user_prompt)
            if input_request is None
            else input_request
        )
        model_responses = []
        while per_request_responses > 0:
            n_responses = min(per_request_responses, 20)
            per_request_responses -= n_responses
            self.generation_config["n"] = n_responses
            reponse = self.completion_with_backoff(
                model=self.model_name,
                messages=input_messages,
                **self.generation_config,
            )
            model_responses.append(reponse)

        return model_responses

    def extract_answers(self, responses: list):
        """Extracting answer from the response of the model."""
        answers = []
        for res in responses:
            answers.extend(choice["message"]["content"] for choice in res["choices"])
        return answers

    def extract_tokens(self, responses: list):
        """Extracting tokens from the responses."""
        completion_tokens = 0
        prompt_tokens = 0
        for res in responses:
            completion_tokens += res["usage"]["completion_tokens"]
            prompt_tokens += res["usage"]["prompt_tokens"]
        return completion_tokens, prompt_tokens

    def extract_response_target_answer(self, responses: list):
        """Extracting the target answer from the responses."""

        prefix = re.escape(self.target_answer_format)
        # 1. extract the string after the answer format
        pattern = rf"{prefix}\s*([^.,\n]+)"

        obtained_targets = []
        for content in responses:
            match = re.search(pattern, content, re.IGNORECASE)

            obtained_targets.append(match.group(1) if match else None)

        return obtained_targets


if __name__ == "__main__":
    chatgpt_api = ChatGPTAPIRequest(
        model_config={"model_name": "gpt-3.5-turbo"}, envs_config=None
    )

    # Define the system message
    system_msg = "You are a helpful assistant who understands data science."

    # Define the user message
    user_msg = 'Create a small dataset about total sales over the last year. The format of the dataset should be a data frame with 12 rows and 2 columns. The columns should be called "month" and "total_sales_usd". The "month" column should contain the shortened forms of month names from "Jan" to "Dec". The "total_sales_usd" column should contain random numeric values taken from a normal distribution with mean 100000 and standard deviation 5000. Provide Python code to generate the dataset, then provide the output in the format of a markdown table.'

    created_messages = chatgpt_api.create_messages(
        textual_user_prompt=user_msg, textual_sys_prompt=system_msg
    )

    response = chatgpt_api.perform_request(request_input=created_messages)
    answer = chatgpt_api.extract_answer(response)
    print("\n")
    print(response)
    print(answer)
