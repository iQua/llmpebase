"""
A vector database for storing the vectors of the text data.

Llmpebase utilizes a third-part library, txtai, to store the vectors of the text data.

See https://neuml.github.io/txtai/embeddings/configuration/ for configuration details.
"""

from txtai.embeddings import Embeddings


class BaseTextVectorDBWorker:
    """
    A base database toward saving vector of the text data.
    """

    def __init__(
        self,
        embedding_model_path: str,
        config: dict = None,
    ):

        # By default, the sentence-transformers/all-MiniLM-L6-v2
        # will be used to get the text embedding
        embedding_model = embedding_model_path

        default_config = {
            "path": embedding_model,
            "content": "sqlite",
            "backend": "torch",
        }

        self.config = config if config is not None else {}
        self.config.update(default_config)
        self.vector_db = Embeddings(self.config)

    def get_uuid(self, queries: list):
        """
        Get the UUID of queries from the database.
        This is generally used to retrieval ids of the queries from the database
        """
        return [result["id"] for result in self.vector_db.search(queries, 1)]


if __name__ == "__main__":
    # A demo to show how to use the vector database in cooperate with the text database

    data = [
        "US tops 5 million confirmed virus cases",
        "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
        "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
        "The National Park Service warns against sacrificing slower friends in a bear attack",
        "Maine man wins $1M from $25 lottery ticket",
        "Make huge profits without work, earn up to $100,000 a day",
    ]
    import os
    from llmpebase.database.generic import BaseDBQuestionRow
    from llmpebase.database.text_db import BaseTextDBWorker

    # Create a text database for the data
    text_db_worker = BaseTextDBWorker(database_path="./test_vb/test_text.db")
    os.makedirs("./test_vb", exist_ok=True)
    text_db_worker.create_tables(
        table_names=["TEST"], row_format=BaseDBQuestionRow.db_row_format()
    )
    rows = [
        BaseDBQuestionRow(
            identity="test-knowledge",
            question=question,
            dataset_name="my_test",
            field="normal",
            category="knowledge",
        )
        for question in data
    ]

    created_uids = text_db_worker.insert_rows(base_rows=rows, table_name="TEST")

    # Create vector database for the data
    vb_worker = BaseTextVectorDBWorker(
        embedding_model_path="sentence-transformers/nli-mpnet-base-v2",
    )

    # Get data ids and data from the text database
    vb_data = [(q_uuid, data[idx]) for idx, q_uuid in created_uids]

    vb_worker.vector_db.index(documents=vb_data)
    vb_worker.vector_db.save(path="test_vb")

    vb_worker.vector_db.load(path="test_vb")

    # Run an embeddings search for each query
    for query in (
        "feel good story",
        "climate change",
        "public health story",
        "war",
        "wildlife",
        "asia",
        "lucky",
        "dishonest junk",
        "Make huge profits without work, earn up to $100,000 a day",
    ):
        # Extract uid of first result
        # search result format: (uid, score)
        print(vb_worker.vector_db.search(query, 1))
        uid = vb_worker.vector_db.search(query, 1)[0]["id"]

        # Print text
        print("query:", query)
        result = vb_worker.vector_db.search(
            f"select id, text from txtai where id = '{uid}'"
        )
        print("result:", result)
