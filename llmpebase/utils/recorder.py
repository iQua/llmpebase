"""
Implementation of recorders for saving outputs to files.
"""

import os
from typing import List
import json

from llmpebase.dataset.data_generic import BaseQASample


class BaseRecorder:
    """
    A base recorder used to save the sample and the outputs.
    """

    def __init__(
        self,
        output_filename: str = "outputs",
        sample_filename: str = "samples",
        statistic_filename: str = "consumption_statistics",
        record_path: str = None,
        record_name: str = None,
    ) -> None:
        self.output_filename = output_filename
        self.sample_filename = sample_filename
        self.statistic_filename = statistic_filename

        self.record_path = os.getcwd() if record_path is None else record_path
        self.record_name = "records" if record_name is None else record_name

        self.record_dir_path = os.path.join(self.record_path, record_name)

        os.makedirs(self.record_dir_path, exist_ok=True)

        self.output_save_path = os.path.join(
            self.record_dir_path, self.output_filename + ".json"
        )
        self.sample_save_path = os.path.join(
            self.record_dir_path, self.sample_filename + ".json"
        )
        self.statistic_save_path = os.path.join(
            self.record_dir_path, self.statistic_filename + ".json"
        )
        # Record of the outputs
        self.outputs: List[dict] = []
        # Record of the id of samples
        self.samples: List[BaseQASample] = []
        # Record of resource consumption statistics
        self.statistics: List[dict] = []

    def add_one_record(self, sample: BaseQASample, output: dict, statistic: dict):
        """Add one record to the recorder."""
        self.outputs.append(output)
        self.samples.append(sample)

        self.statistics.append(statistic)

    def save_records(self):
        """Saving the records to the disk."""

        with open(self.output_save_path, "w", encoding="utf-8") as file:
            json.dump(self.outputs, file)

        with open(self.sample_save_path, "w", encoding="utf-8") as file:
            json.dump(self.samples, file)

        with open(self.statistic_save_path, "w", encoding="utf-8") as file:
            json.dump(self.statistics, file)
