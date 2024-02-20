"""
The implementation of the database for Experiences.

It deserves to emphasize that the variables of the experiences should be 
the same as the columns of the database in terms of the name and the order.
"""

import os
import sqlite3
from typing import List, Dict

from txtai.embeddings import Embeddings

from experience_generic import BaseExperience

from llmpebase.dataset.data_generic import BaseQASample

# The positive and negative here derives from the concept of
# Triplet Loss.
# That is to say, when the experience is to be used, some
# are positive and some are negative, both contributing to
# the learning.
ExperienceTypes = ["Positive", "Negative"]


class ExperienceDB:
    """
    A database for experiences in which there are two tables corresponding to
    reasoning and verifications, respectively.

    Args:
        database_path: The path to the database.
    """

    def __init__(
        self,
        database_path: str,
    ):

        # Create a new database when there is no one
        self.database_path = database_path

        self.experience_mode = ["Reasoning", "Verification"]

        # Get the table elements for the corresponding experience type
        self.table_elements = BaseExperience.database_format()

    def initialize_db(self):
        """Initialize the database."""

        # Ensure that an empty database is created
        db_connector = sqlite3.connect(self.database_path)
        cursor = db_connector.cursor()

        # SQL statement to create tables
        for experience_mode in self.experience_mode:
            table_command = self.create_db_table(table_name=experience_mode)
        cursor.execute(table_command)

        # Commit the changes
        db_connector.commit()
        # Close the cursor and the connection
        cursor.close()
        db_connector.close()

    def create_db_table(self, table_name: str):
        """Create a table in the database."""

        # Convert the table elements to be the database format
        columns = ", ".join(
            [
                f"{col_name} {data_type}"
                for col_name, data_type in self.table_elements.items()
            ]
        )
        return f"CREATE TABLE IF NOT EXISTS {table_name} ({columns});"

    def insert_experiences(self, experiences: List[BaseExperience], table_name: str):
        """
        Direct insert experiences to the table 'table_name' of the database.

        Note that the experiences are inserted into the table without being
        ordered based on priority score. The main reason is that ordering
        experiences in the database is unnecessary and it is better to
        retrieve experiences based on the priority score, as what we do
        in 'retrieve_experiences'.
        """

        # Connect to the database
        db_connector = sqlite3.connect(self.database_path)
        cursor = db_connector.cursor()

        # Convert the experience to the database format,
        # i.e., a tuple in which each item contain the value
        # of the corresponding column in the table
        db_experiences = [experience.to_database_data() for experience in experiences]
        placeholders = ", ".join(["?"] * len(db_experiences))
        columns = ", ".join(self.table_elements.keys())

        cmd = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        cursor.executemany(cmd, db_experiences)

        # Commit the insert operations
        db_connector.commit()

        # Close the cursor and the connection
        cursor.close()
        db_connector.close()

    def retrieve_experiences(
        self, table_name: str, n_shots: int = 1, flag="priority_score"
    ):
        """
        Retrieve n_shots of experiences from the table 'table_name' of the database.

        Only when retrieving the experiences, the experiences are ordered.
        """

        # Connect to the database
        db_connector = sqlite3.connect(self.database_path)
        cursor = db_connector.cursor()

        # Retrieve the top 'n_shots' experiences ordered by highest priority
        select_sql = f"SELECT * FROM {table_name} ORDER BY {flag} ASC LIMIT ?"
        cursor.execute(select_sql, (n_shots,))
        # Fetch the db experiences which are maintained as a list of tuples
        # in which each tuple contains the values of the experience -- table
        # columns
        return [
            BaseExperience.from_database_data(row_item)
            for row_item in cursor.fetchall()
        ]


class LLMExperienceDBManager:
    """
    A holder to holding all different experiences learned by the LLM.
    """

    def __init__(self, model_config: dict) -> None:

        # By default, the sentence-transformers/all-MiniLM-L6-v2
        # will be used to get the text embedding
        embedding_model = model_config.get("experience_memorization", {}).get(
            "question_db_embedding_model", {}
        )
        self.question_db = Embeddings(
            {
                "path": embedding_model,
                "content": "sqlite",
                "backend": "torch",
            }
        )

    def get_experience_database(self, location: str, db_name: str):
        """Get the database for the experience."""
        experience_style = experience_style.title()
        database_path = os.path.join(location, db_name)
        return ExperienceDB(database_path=database_path)

    def record_questions(
        self,
        samples: List[BaseQASample],
    ):
        """Build embeddings for questions."""

        # Index data in the upsert version
        question_ids = self.get_question_id(samples)
        questions = [sample["question"] for sample in samples]
        self.question_db.upsert(
            (question_id, question, None)
            for question_id, question in zip(question_ids, questions)
        )

    def record_experiences(
        self,
        locations: List[str],
        batch_samples: List[BaseQASample],
        batch_experiences: List[Dict[str, List[str]]],
        batch_accuracy: float,
    ):
        """Insert the experiences to the database."""
        for idx, experiences in enumerate(batch_experiences):
            location = locations[idx]
            sample = batch_samples[idx]

            question_id = self.get_question_id(sample=sample)
            # Create the identity of the experience
            # here we do not perform cluster for the experiences,
            # thus the cluster_id is set to 0
            experience_id = self.create_experience_id(
                sample=sample, cluster_id=0, experience_style=style
            )
            # Insert the experiences to the database
            # First, get the specific database and open it
            database = self.get_experience_database(
                location=location, db_name="ExperienceDB.db"
            )

            reasoning_experiences = experiences["Reasoning"]
            verify_experiences = experiences["Verification"]


    def create_db_experience(self, sample: BaseQASample, experiences: List[str], experience_style: str):
        """ Create the experience to be inserted to the database."""
        question_id = self.get_question_id(sample=sample)
        experience_ids = self.create_experience_id(
                sample=sample, cluster_id=0, experience_style=experience_style
            )
        return [BaseExperience(question_identity=question_id, identity=experience_id, experience = experience, experience_style=experience_style, priority_score=0, historical_priorities=[0]) for experience_id, experience in zip(experience_ids, experiences)]


    def create_experience_id(self, sample: BaseQASample, cluster_id: str, experience_style: str):
        """Create the experience id."""
        question_id = self.get_question_id(sample)
        return f"{question_id}-{cluster_id}-{experience_style}"

    def get_question_id(self, sample: BaseQASample):
        info = sample["auxiliary"]["sample_info"]
        return f"{info["sample_dataset"]}-{info["sample_field"]}-{info["sample_problem"]}-{info["sample_id"]}"

    def search_question_cluster(
        self,
        questions: List[str],
        group_folder: str,
        experience_style: str,
        experience_mode: str,
    ):
        """Search the experience cluster for the questions."""
        # Get the embeddings of the questions
        q_embeddings = self.embedder(questions)

        # Get the database of the group
        database = self.get_experience_database(
            group_folder, experience_style, experience_mode
        )
        # There may have several tables, each corresponding to a cluster, in the database
