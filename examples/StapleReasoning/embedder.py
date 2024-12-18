"""
An embedding module used by the p-RAR.
"""

from typing import List

import numpy as np
from scipy.spatial import distance
from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator, EmbeddingDistance

from dotenv import load_dotenv


class GPTEmbedder(object):
    """An embedder for the GPT model from the OpenAI."""

    def __init__(self, model_config: dict) -> None:
        auth_env_path = model_config["authorization_path"]
        # there must have a .env file containing keywords
        # OPENAI_ORGAN_KEY and OPENAI_API_KEY
        load_dotenv(auth_env_path)

        embedding_config = model_config["embedder"]

        # There are three options of the openAI emebdding
        # text-embedding-3-small, $0.02 / 1M tokens
        # text-embedding-3-large, $0.13 / 1M tokens
        # ada v2, $0.10 / 1M tokens
        model_name = embedding_config["model_name"]

        self.distance_metric = embedding_config["distance_metric"]

        self.embeddings_model = OpenAIEmbeddings(model=model_name)

    def get_distance_scores(self, query: str, documents: List[str]):
        """
        Prepare the query and documents.

        This functions is invalid.
        """
        evaluator = load_evaluator(
            "pairwise_embedding_distance",
            distance_metric=EmbeddingDistance(self.distance_metric),
            embeddings=self.embeddings_model,
        )
        return [
            evaluator.evaluate_string_pairs(
                prediction=query,
                prediction_b=doc,
            )
            for doc in documents
        ]

    def get_neighbors(self, query: str, documents: List[str], num_neighbors: int):
        """Perform the embedding of the text."""
        # Perform the embedding and get embeddings as lists
        # with shape (n_documents, embedding_size) and (embedding_size,)
        doc_embeddings = self.embeddings_model.embed_documents(documents)
        query_embedding = self.embeddings_model.embed_query(query)

        # Compute the distance
        query_embedding = np.array(query_embedding).reshape(1, -1)
        # A matrix showing the distance
        # with shape [n_documents,]
        distances = distance.cdist(
            query_embedding, doc_embeddings, self.distance_metric
        )[0]
        # From smallest to largest, get top k
        top_k_indices = np.argsort(distances)[:num_neighbors]
        return top_k_indices, distances[top_k_indices]
