"""
Generic components for the experience.
"""

import json
from collections import OrderedDict
from typing import List, get_type_hints
from dataclasses import dataclass


from transformers.utils import ModelOutput as FieldFrozenContainer


@dataclass
class BaseDBRow(FieldFrozenContainer):
    """A base term for the database."""

    def to_database_data(self):
        """Convert the values of the row to the data to be stored in the database."""
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
        Format the experience to be stored in the database.

        Note that even thought for Python 3.7 and later, the order of the
        items in the dictionary is preserved, we still explicitly use
        OrderedDict to show that the order must be preserved.
        Note that 'db_identity' is added here to ensure that database will
        have a unique primary key.
        """
        return OrderedDict(
            [
                ("db_identity", "TEXT PRIMARY KEY"),
            ]
        )


@dataclass
class BaseExperience(FieldFrozenContainer):
    """
    A base experience holding the necessary terms.
    """

    # The content of the experience.
    experience: str

    # The style of the experience
    # It should be Reasoning or Verification
    style: str = None

    # The mode of the experience
    # It should be Positive or Negative
    mode: str = None

    # The priority score of the experience.
    priority_score: float = None


@dataclass
class BaseDBQuestion(BaseDBRow):
    """
    A base experience holding the necessary terms.
    """

    # The identity of the question.
    # <dataset>-<field>-<category>-<sample_id>.
    identity: str = None

    dataset: str = None
    field: str = None
    category: str = None
    sample_id: str = None
    question: str = None

    @staticmethod
    def db_row_format():
        """
        Format the experience to be stored in the database.

        Note that even thought for Python 3.7 and later, the order of the
        items in the dictionary is preserved, we still explicitly use
        OrderedDict to show that the order must be preserved.
        Note that 'db_identity' is added here to ensure that database will
        have a unique primary key.
        """
        return OrderedDict(
            [
                ("db_identity", "TEXT PRIMARY KEY"),
                ("identity", "TEXT NOT NULL"),
                ("dataset", "TEXT NOT NULL"),
                ("field", "TEXT NOT NULL"),
                ("category", "TEXT NOT NULL"),
                ("sample_id", "TEXT"),
                ("question", "TEXT"),
            ]
        )


@dataclass
class BaseDBExperience(BaseDBRow):
    """
    A base experience for the database.
    """

    # The identity of the experience.
    # This should be unique for each experience and the format
    # is general: <dataset>-<field>-<category>-<sample_id>-<cluster_id>-<style>-<mode>.
    # where the style can be Reasoning or Verification
    identity: str = None

    # The identity of the question.
    # <dataset>-<field>-<category>-<sample_id>.
    question_identity: str = None

    # The content of the experience.
    experience: str = None

    # The style of the experience
    # It should be Reasoning or Verification
    style: str = None

    # The mode of the experience
    # It should be Positive or Negative
    mode: str = None

    # The priority score of the experience.
    priority_score: float = None

    # Historical priorities
    historical_priorities: List[float] = None

    @staticmethod
    def db_row_format():
        """
        Format the experience to be stored in the database.

        Note that even thought for Python 3.7 and later, the order of the
        items in the dictionary is preserved, we still explicitly use
        OrderedDict to show that the order must be preserved.
        Note that 'db_identity' is added here to ensure that database will
        have a unique primary key.
        """
        return OrderedDict(
            [
                ("db_identity", "TEXT PRIMARY KEY"),
                ("identity", "TEXT NOT NULL"),
                ("question_identity", "TEXT NOT NULL"),
                ("experience", "TEXT NOT NULL"),
                ("style", "TEXT NOT NULL"),
                ("mode", "TEXT"),
                ("priority_score", "REAL"),
                ("historical_priorities", "TEXT"),
            ]
        )
