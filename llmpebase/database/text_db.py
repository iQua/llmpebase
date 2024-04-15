"""
A normal database used to save text.
In Llmpebase, the normal database used is the sqlite3 database.
"""

import sqlite3
from typing import List, Dict
from collections import defaultdict

from llmpebase.database.generic import BaseDBRow
from llmpebase.database.generic import create_uuids


class BaseTextDBWorker:
    """
    A base database toward saving text data.
    """

    def __init__(self, database_path: str, uuid_method: str = "uuid5"):
        # Create a new database when there is no one
        self.database_path = database_path
        # Define the method to generate the UUID
        self.uuid_method = uuid_method

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
        cursor.close()
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

        # Create the ids of the rows when they are not available
        created_uuids = create_uuids(base_rows)
        for idx, row_uuid in created_uuids:
            base_rows[idx].UUID = row_uuid

        # Check if the row exists
        for idx, row_data in enumerate(base_rows):
            cmd = f"SELECT COUNT(*) FROM {table_name} WHERE UUID = ? AND identity = ?"
            cursor.execute(cmd, (row_data.UUID, row_data.identity))
            if cursor.fetchone()[0] > 0:
                # Report an error when the row exists
                raise ValueError(
                    f"Row with UUID {row_data.UUID} and identity {row_data.identity} exists in table {table_name}"
                )

        # Convert the experience to the database format,
        # i.e., a tuple in which each item contain the value
        # of the corresponding column in the table
        db_rows = [row_data.to_database_data() for row_data in base_rows]

        # Get the rows of the experience to be saved
        element_names = base_rows[0].keys()
        placeholders = ", ".join(["?"] * len(element_names))
        columns = ", ".join(element_names)
        cmd = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        cursor.executemany(cmd, db_rows)

        # Update the database identity of the rows
        cursor.execute(f"UPDATE {table_name} set RowIDX = 'RIDX' || rowid")

        # Commit the insert operations
        db_connector.commit()

        # Close the cursor and the connection
        cursor.close()
        db_connector.close()

        return created_uuids

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
        row_uuid = base_row.UUID
        row_identity = base_row.identity

        # Connect to the database
        cursor = db_connector.cursor()
        # Check if a matching row exists
        cmd = f"SELECT COUNT(*) FROM {table_name} WHERE UUID = ? AND identity = ?"
        cursor.execute(cmd, (row_uuid, row_identity))

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

    def retrieve_rows(
        self,
        table_name: str,
        n_shots: int = 1,
        flag="priority_score",
    ):
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

        return cursor.fetchall()

    def check_row_existence(self, table_name: str, row_data: BaseDBRow):
        """Check if the row exists in the table."""
        # Connect to the database
        db_connector = sqlite3.connect(self.database_path)
        cursor = db_connector.cursor()
        cmd = f"SELECT COUNT(*) FROM {table_name} WHERE UUID = ? AND identity = ?"
        cursor.execute(cmd, (row_data.UUID, row_data.identity))
        # Fetch the result, which is the count of rows matching the criteria
        count = cursor.fetchone()[0]

        return count > 0

    def count_tables(
        self,
        by_columns: List[str],
    ) -> Dict[str, Dict[str, int]]:
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


if __name__ == "__main__":
    # Works with a list, dataset or generator
    from llmpebase.database.generic import BaseDBQuestionRow
    import os

    data = [
        "US tops 5 million confirmed virus cases",
        "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
        "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
        "The National Park Service warns against sacrificing slower friends in a bear attack",
        "Maine man wins $1M from $25 lottery ticket",
        "Make huge profits without work, earn up to $100,000 a day",
    ]
    text_db_worker = BaseTextDBWorker(database_path="./test_vb/test_text.db")
    os.makedirs("./test_vb", exist_ok=True)
    text_db_worker.create_tables(
        table_names=["TEST2"], row_format=BaseDBQuestionRow.db_row_format()
    )

    rows = [
        BaseDBQuestionRow(
            identity="test-knowledge",
            question=question,
            dataset_name="my_test",
            field="test",
            category="knowledge",
        )
        for question in data
    ]

    created_uids = text_db_worker.insert_rows(base_rows=rows, table_name="TEST2")
    print(created_uids)
