"""
A series of LLMs for the text embedding.
"""

from typing import List

import openai
import numpy as np
from tenacity import retry, wait_random_exponential, stop_after_attempt


class OpenAIEmbedder:
    """A class to create the OpenAI Embedder."""

    def __init__(self, embed_model_config: dict):
        self.model_name = (
            embed_model_config["model_name"]
            if "model_name" in embed_model_config
            else "text-embedding-3-small"
        )

        assert self.model_name in [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]

    # Retry up to 6 times with exponential backoff, starting at 1 second and maxing out at 20 seconds delay
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def forward(self, batch_texts: List[str]):
        """Forward the model to perform a request."""
        responses = openai.embeddings.create(input=batch_texts, model=self.model_name)
        # match completions to prompts by index
        embeddings = [0] * len(batch_texts)
        for choice in responses.choices:
            embeddings[choice.index] = choice["data"][0]["embedding"]

        # Stack the embeddings into a 2D array
        return np.stack(embeddings)
