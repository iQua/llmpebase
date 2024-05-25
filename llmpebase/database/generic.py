"""
Implementation of generic components of the database.

"""

import json
from collections import OrderedDict
from typing import List, get_type_hints
from dataclasses import dataclass

from txtai.embeddings import AutoId

from transformers.utils import ModelOutput as FieldFrozenContainer


@dataclass
class BaseDBRow(FieldFrozenContainer):
    """
    A base row holding the necessary terms.
    """

    # The unique identifier of the row
    # This is a UUID string used across different databases.
    UUID: str = "NULL"

    # The identity of the row content
    # This is different from the primary key, RowID which presents
    # the row index.
    # For example, identity may contain necessary information to be related
    # with other rows.
    identity: str = None

    def to_database_data(self):
        """
        Convert the values of the row to the data to be stored in the database."""
        db_columns = list()
        type_hints = get_type_hints(type(self))
        # Extract variable values from the elements
        # Convert the list and dict to JSON string
        # Eventually make them a database format
        for key, value in self.items():
            # Check whether the type hint is a list or dict
            if getattr(type_hints[key], "__origin__", None) is list:
                # Serialize list to JSON string
                # Replace the value in the element
                value = json.dumps(value)

            db_columns.append(value)

        return tuple(db_columns)

    def from_database_data(self, db_columns: tuple):
        """
        Assign the elements from the one retrieved from the database.

        Note that the value order in the db_columns should be the same
        as the order of the fields.
        """
        fields = list(self.keys())
        type_hints = get_type_hints(type(self))
        for idx, key in enumerate(fields):
            value = db_columns[idx]
            if getattr(type_hints[key], "__origin__", None) is list:
                value = json.loads(value)

            setattr(self, key, value)

    @staticmethod
    def db_row_format():
        """
        Format the row to be stored in the database.

        Note that even thought for Python 3.7 and later, the order of the
        items in the dictionary is preserved, we still explicitly use
        OrderedDict to show that the order must be preserved.
        Note that 'RowIDX' is added here to ensure that database will
        have a unique primary key.
        """
        return OrderedDict(
            [
                ("RowIDX", "TEXT PRIMARY KEY"),
                ("UUID", "TEXT NOT NULL"),
                ("identity", "TEXT NOT NULL"),
            ]
        )


@dataclass
class BaseDBQuestionRow(BaseDBRow):
    """
    A row holding the necessary terms of a question.
    """

    # The identity of the question.
    # <dataset_name>-<field>-<category>-<sample_id>.

    question: str = None
    dataset_name: str = None
    field: str = None
    category: str = None
    sample_id: str = None

    def set_default_identity(self):
        """
        Create the identity of the question.
        """
        if self.identity is None:
            self.identity = (
                f"{self.dataset_name}-{self.field}-{self.category}-{self.sample_id}"
            )
        return self.identity

    @staticmethod
    def db_row_format():
        """
        Format the items to be stored in the question row.
        """
        basic_items = BaseDBRow.db_row_format()
        basic_items.update(
            OrderedDict(
                [
                    ("question", "TEXT NOT NULL"),
                    ("dataset_name", "TEXT"),
                    ("field", "TEXT"),
                    ("category", "TEXT"),
                    ("sample_id", "TEXT"),
                ]
            )
        )
        return basic_items


def create_uuids(sequences: List[BaseDBRow], method="uuid5"):
    """
    Generate UUIDs for the sequence.

    uuid1: This generates a UUID using a host ID, sequence number, and the current time.
    uuid3 and uuid5: These generate a UUID using the MD5 (for uuid3) or SHA1 (for uuid5) hash of a namespace identifier and a name.
    uuid4: This generates a random UUID.

    Note that uuid1 and uuid4 are not deterministic while uuid3 and uuid5 are deterministic, meaning that uuid3 and uuid5 will always generate the same ids for the same sequence.
    """
    id_generator = AutoId(method=method)
    return [
        (idx, id_generator(seq))
        for idx, seq in enumerate(sequences)
        if seq.UUID == "NULL"
    ]
