"""
The implementation of the database for Experiences.

It deserves to emphasize that the variables of the experiences should be 
the same as the columns of the database in terms of the name and the order.
"""

import os
import sqlite3
import json
from typing import List, Dict
from collections import defaultdict

from txtai.embeddings import Embeddings

from experience_generic import (
    BaseDBRow,
    BaseExperience,
    BaseDBExperience,
    BaseDBQuestion,
)

from llmpebase.dataset.data_generic import BaseQASample


ExperienceMode = ["Positive", "Negative"]
ExperienceStyle = ["Reasoning", "Verification"]


class BaseDBWorker:
    """
    A base implementation of the database.
    """

    def __init__(self, database_path: str):
        # Create a new database when there is no one
        self.database_path = database_path

    def create_db_table(self, table_name: str, row_format: List[str]):
        """Create a table in the database."""

        # Convert the table rows to be the database format
        columns = ", ".join(
            [f"{col_name} {data_type}" for col_name, data_type in row_format.items()]
        )
        return f"CREATE TABLE IF NOT EXISTS {table_name} ({columns});"

    def create_tables(self, table_names: str, row_format: List[str]):
        """Initialize the database."""
        # Connect to the database
        db_connector = sqlite3.connect(self.database_path)
        cursor = db_connector.cursor()

        # Create the table
        for table_name in table_names:
            cursor.execute(self.create_db_table(table_name, row_format=row_format))

        # Commit the changes
        db_connector.commit()

        # Close the connection
        db_connector.close()

    def insert_rows(self, base_rows: List[BaseDBRow], table_name: str):
        """
        Direct insert rows to the table 'table_name' of the database.

        Note that the base_rows are inserted into the table without being
        ordered based on priority score. The main reason is that ordering
        base_rows in the database is unnecessary and it is better to
        retrieve base_rows based on the priority score, as what we do
        in 'retrieve_base_rows'.
        """

        # Connect to the database
        db_connector = sqlite3.connect(self.database_path)
        cursor = db_connector.cursor()

        # Convert the experience to the database format,
        # i.e., a tuple in which each item contain the value
        # of the corresponding column in the table
        db_experiences = [base_exp.to_database_data() for base_exp in base_rows]

        # Get the rows of the experience to be saved
        element_names = base_rows[0].keys()
        placeholders = ", ".join(["?"] * len(element_names))
        columns = ", ".join(element_names)
        cmd = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        cursor.executemany(cmd, db_experiences)

        # Update the database identity of the experiences
        cursor.execute(f"UPDATE {table_name} set db_identity = 'OID' || rowid")

        # Commit the insert operations
        db_connector.commit()

        # Close the cursor and the connection
        cursor.close()
        db_connector.close()

    def update_one_row(
        self,
        db_connector: sqlite3.Connection,
        base_row: BaseDBRow,
        table_name: str,
    ):
        """
        Update the experiences in the table 'table_name' of the database.

        A new experience is regarded as the same as the one in the table when
        they have the same question identity, identity, and experience.
        """
        identity = base_row.identity

        # Connect to the database
        cursor = db_connector.cursor()
        # Check if a matching row exists
        cmd = f"SELECT identity FROM {table_name} WHERE identity = ?"
        cursor.execute(cmd, (identity,))

        result = cursor.fetchone()

        if not result:
            # Row does not exist, insert a new one
            self.insert_rows([base_row], table_name)

        # Commit the changes
        db_connector.commit()

    def update_rows(self, base_rows: List[BaseDBRow], table_name: str):
        """
        Update the rows in the table 'table_name' of the database.
        """

        # Connect to the database
        db_connector = sqlite3.connect(self.database_path)

        for db_row in base_rows:
            self.update_one_row(db_connector, db_row, table_name)

        # Close the connection
        db_connector.close()

    def retrieve_rows(self, table_name: str, n_shots: int = 1, flag="priority_score"):
        """
        Retrieve n_shots of rows from the table 'table_name' of the database.
        Only when retrieving the rows, the rows are ordered.
        """

        # Connect to the database
        db_connector = sqlite3.connect(self.database_path)
        cursor = db_connector.cursor()

        # Retrieve the top 'n_shots' rows ordered by highest priority
        select_sql = f"SELECT * FROM {table_name} ORDER BY {flag} ASC LIMIT ?"
        cursor.execute(select_sql, (n_shots,))
        # Fetch the db rows which are maintained as a list of tuples
        # in which each tuple contains the values of the experience -- table
        # columns
        return [
            BaseDBRow.from_database_data(row_item) for row_item in cursor.fetchall()
        ]

    def count_tables(
        self,
        by_columns: List[str],
    ):
        """Count the number of rows in the table."""

        # Connect to the database
        db_connector = sqlite3.connect(self.database_path)
        cursor = db_connector.cursor()

        # Query to get the names of all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        # Fetch all results from the query
        table_names = [table[0] for table in cursor.fetchall()]

        by_columns = (
            by_columns * len(table_names) if len(by_columns) == 1 else by_columns
        )

        # Count the number of rows in the table
        db_statistics = defaultdict(dict)
        for idx, table_name in enumerate(table_names):
            column_name = by_columns[idx]
            query = f"SELECT {column_name}, COUNT(*) as count FROM {table_name} GROUP BY {column_name}"
            cursor.execute(query)
            for key, value in cursor.fetchall():
                db_statistics[table_name][key] = value

        # Close the connection
        db_connector.close()

        return db_statistics


class ExperienceDBWorker(BaseDBWorker):
    """
    A database for experiences in which there are two tables corresponding to
    reasoning and verifications, respectively.

    Args:
        database_path: The path to the database.
    """

    def update_one_row(
        self,
        db_connector: sqlite3.Connection,
        base_row: BaseDBExperience,
        table_name: str,
    ):
        """
        Update the experiences in the table 'table_name' of the database.

        A new experience is regarded as the same as the one in the table when
        they have the same question identity, identity, and experience.
        """
        identity = base_row.identity
        question_id = base_row.question_identity
        experience = base_row.experience
        new_priority = base_row.priority_score

        # Connect to the database
        cursor = db_connector.cursor()
        # Check if a matching row exists
        cmd = f"SELECT priority_score, historical_priorities FROM {table_name} WHERE identity = ? AND question_identity = ? AND experience = ?"
        cursor.execute(cmd, (identity, question_id, experience))

        result = cursor.fetchone()

        if result:
            # Row exists, perform the merge
            old_priority, history_priority = result
            combined_priority = old_priority + new_priority
            # Deserialize the history_priority JSON string into a list
            history_priority_list = json.loads(history_priority)
            history_priority_list.append(old_priority)
            history_priority = json.dumps(history_priority_list)

            cmd = """UPDATE items SET priority_score = ?, historical_priorities = ? WHERE identity = ? AND question_identity = ? AND experience = ?"""
            cursor.execute(
                cmd,
                (
                    combined_priority,
                    history_priority,
                    identity,
                    question_id,
                    experience,
                ),
            )
        else:
            # Row does not exist, insert a new one
            self.insert_rows([base_row], table_name)

        # Commit the changes
        db_connector.commit()


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

    def get_question_database(self, location: str, db_name: str):
        """Get the database for the questions."""
        database_path = os.path.join(location, db_name)
        question_db = BaseDBWorker(database_path=database_path)
        question_db.create_tables(
            table_names=["Questions"], row_format=BaseDBQuestion.db_row_format()
        )
        return question_db

    def get_experience_database(self, location: str, db_name: str):
        """Get the database for the experience."""
        database_path = os.path.join(location, db_name)
        experience_db = ExperienceDBWorker(database_path=database_path)
        experience_db.create_tables(
            table_names=ExperienceStyle, row_format=BaseDBExperience.db_row_format()
        )
        return experience_db

    def record_questions(
        self,
        location: str,
        samples: List[BaseQASample],
    ):
        """Build embeddings for questions."""

        database = self.get_question_database(
            location=location, db_name="QuestionDB.db"
        )
        db_questions = [self.create_db_question(sample=sample) for sample in samples]

        database.update_rows(db_questions, "Questions")
        # Count the current rows
        return database.count_tables(["dataset"])

    def record_experiences(
        self,
        location: List[str],
        sample: BaseQASample,
        experiences: Dict[str, List[BaseExperience]],
    ):
        """Insert the experiences to the database."""

        # Get the experience database
        database = self.get_experience_database(
            location=location, db_name="ExperienceDB.db"
        )

        # Add experiences to the database
        for exp_style in ExperienceStyle:
            # Create the experiences for the database purpose
            style_experiences = self.create_db_experience(
                sample, experiences[exp_style]
            )
            # Insert the experiences to the database
            database.update_rows(style_experiences, exp_style)

        # Count the current rows
        return database.count_tables()

    def create_db_experience(
        self, sample: BaseQASample, base_experiences: List[BaseExperience]
    ):
        """Create the experience to be inserted to the database."""
        question_id = self.get_question_id(sample=sample)
        # Set the default cluster_id to 0 as there is no clustering
        # in the current version of experiences
        experience_ids = [
            self.create_experience_id(
                sample=sample,
                cluster_id=0,
                style=base_exp.style,
                mode=base_exp.mode,
            )
            for base_exp in base_experiences
        ]
        return [
            BaseDBExperience(
                identity=experience_id,
                question_identity=question_id,
                experience=base_exp.experience,
                style=base_exp.style,
                mode=base_exp.mode,
                priority_score=base_exp.priority_score,
                historical_priorities=[],
            )
            for experience_id, base_exp in zip(experience_ids, base_experiences)
        ]

    def create_db_question(self, sample: BaseQASample):
        """Create the question to be inserted to the database."""
        question_id = self.get_question_id(sample=sample)
        sample_info = sample["auxiliary"]["sample_info"]
        return BaseDBQuestion(
            identity=question_id,
            dataset=sample_info["sample_dataset"],
            field=sample_info["sample_field"],
            category=sample_info["sample_problem"],
            sample_id=sample_info["sample_id"],
            question=sample["question"],
        )

    def create_experience_id(
        self, sample: BaseQASample, cluster_id: str, style: str, mode: str
    ):
        """Create the experience id."""
        question_id = self.get_question_id(sample)
        return f"{question_id}-{cluster_id}-{style}-{mode}"

    def get_question_id(self, sample: BaseQASample):
        info = sample["auxiliary"]["sample_info"]
        dataset = info["sample_dataset"]
        field = info["sample_field"]
        category = info["sample_problem"]
        sample_id = info["sample_id"]
        return f"{dataset}-{field}-{category}-{sample_id}"
