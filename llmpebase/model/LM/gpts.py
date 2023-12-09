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

import os
import logging
from typing import List

from openai import OpenAI
from dotenv import load_dotenv

from llmpebase.model.LM import base


class GPTAPIRequest(base.BaseLlmRequest):
    """A class to forward the ChatGPT model with API of OPENAI."""

    def __init__(self, model_config: dict) -> None:
        super().__init__(model_config)

        assert self.model_name in ["gpt-3.5-turbo", "gpt-4"]

        auth_env_path = model_config["authorization_path"]
        # there must have a .env file containing keywords
        # OPENAI_ORGAN_KEY and OPENAI_API_KEY
        load_dotenv(auth_env_path)

        # Define the client
        self.client = None

    def configuration(self):
        """Configure the GPT model."""
        super().configuration()

        generation_settings = self.model_config["generation_settings"]

        # Set the hyper-parameters for the generation
        # 1. 'n' allows one to specify how many responses to be received
        # for each prompt
        self.generation_config["n"] = 1
        self.generation_config["stop"] = None

        self.generation_config.update(generation_settings)

        # Define the client
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        logging.info("Connected to a OPENAI client.")

    def completion_with_backoff(self, **kwargs):
        """Completing the forward within the OPENAI."""
        # Increase the number of requests
        self.num_requests += 1
        return self.client.chat.completions.create(**kwargs)

    def create_format_input(self, user_prompt: str, **kwargs):
        """Create messages for GPT-4 to be used for forwarding."""
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

    def forward(
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
            self.create_format_input(user_prompt, **kwargs)
            if input_request is None
            else input_request
        )
        model_responses = []
        while per_request_responses > 0:
            n_responses = min(per_request_responses, 20)
            per_request_responses -= n_responses
            self.generation_config["n"] = n_responses
            output = self.completion_with_backoff(
                model=self.model_name,
                messages=input_messages,
                **self.generation_config,
            )

            model_responses.append(output)
        # Compute the resources
        self.compute_costs(input_messages, model_responses)

        return model_responses

    def read_response_contents(self, responses: list):
        contents = []
        for res in responses:
            contents.extend(choice.message.content for choice in res.choices)
        return contents

    def compute_costs(self, input_messages: dict, responses: list):
        # Statistics the number of prompt tokens

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
        """ChapGPT has the upper bound of the request rate."""
        return True


if __name__ == "__main__":
    chatgpt_api = GPTAPIRequest(model_config={"model_name": "gpt-3.5-turbo"})

    # Define the system message
    system_msg = "You are a helpful assistant who understands data science."

    # Define the user message
    user_msg = 'Create a small dataset about total sales over the last year. The format of the dataset should be a data frame with 12 rows and 2 columns. The columns should be called "month" and "total_sales_usd". The "month" column should contain the shortened forms of month names from "Jan" to "Dec". The "total_sales_usd" column should contain random numeric values taken from a normal distribution with mean 100000 and standard deviation 5000. Provide Python code to generate the dataset, then provide the output in the format of a markdown table.'

    created_messages = chatgpt_api.create_format_input(
        user_prompt=user_msg, sys_prompt=system_msg
    )

    response = chatgpt_api.forward(request_input=created_messages)
    content = chatgpt_api.extract_response_contents(response)
    print("\n")
    print(response)
    print(content)
