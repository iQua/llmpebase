"""
Generic components for the experience.
"""

import json
from collections import OrderedDict
from typing import List, get_type_hints
from dataclasses import dataclass


from transformers.utils import ModelOutput as FieldFrozenContainer


@dataclass
class BaseExperience(FieldFrozenContainer):
    """
    A base experience holding necessary items without the specific
    content.
    """

    # The identity of the question.
    # <dataset>-<field>-<category>-<sample_id>.
    question_identity: str = None

    # The identity of the experience.
    # This should be unique for each experience and the format
    # is general: <dataset>-<field>-<category>-<sample_id>-<cluster_id>-<type>.
    # where the type can be Reasoning or Verification
    identity: str = None

    # The content of the experience.
    experience: str

    # The style of the experience
    # It should be Reasoning or Verification
    experience_style: str = None

    # The mode of the experience
    # It should be Positive or Negative
    mode: str = None

    # The priority score of the experience.
    priority_score: float = None

    # Historical priorities
    historical_priorities: List[float] = None

    def to_database_data(self):
        """Convert the experience to the data to be stored in the database."""
        database_experience = list()
        type_hints = get_type_hints(type(self))
        # Extract variable values from the experience
        # Convert the list and dict to JSON string
        # Eventually make them a database format
        for key, value in self.keys():
            # Check whether the type hint is a list or dict
            if getattr(type_hints[key], "__origin__", None) in [list, dict]:
                # Serialize list to JSON string
                # Replace the value in the experience
                value = json.dumps(value)
            database_experience.append(value)

        return tuple(database_experience)

    def from_database_data(self, db_experience: tuple):
        """
        Assign the experience from the one retrieved from the database.

        Note that the value order in the db_experience should be the same
        as the order of the fields in the experience.
        """
        fields = list(self.keys())
        type_hints = get_type_hints(type(self))
        for idx, key in enumerate(fields):
            value = db_experience[idx]
            if getattr(type_hints[key], "__origin__", None) in [list, dict]:
                value = json.loads(value)

            setattr(self, key, value)

    @staticmethod
    def database_format():
        """
        Format the experience to be stored in the database.

        Note that even thought for Python 3.7 and later, the order of the
        items in the dictionary is preserved, we still explicitly use
        OrderedDict to show that the order must be preserved.
        """
        return OrderedDict(
            [
                ("identity", "TEXT PRIMARY KEY"),
                ("question_identity", "TEXT NOT NULL"),
                ("experience", "TEXT NOT NULL"),
                ("experience_style", "TEXT NOT NULL"),
                ("mode", "TEXT"),
                ("priority_score", "REAL"),
                ("historical_priorities", "TEXT"),
            ]
        )
