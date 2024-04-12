"""
Implementation of recorders for saving outputs to files.
"""

import os
import json
import glob

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

    def get_recorded_names(self):
        """Get the indexes of the records."""
        pattern = f"{self.record_dir_path}/{self.output_filename}_*.json"

        # Use glob to find files matching the pattern
        exist_records = glob.glob(pattern)

        record_names = [
            record.split("_")[-1].split(".json")[0] for record in exist_records
        ]

        # Order the indexes
        record_names.sort()

        # We did not include the latest record as it may not be completed
        return record_names[:-1]

    def get_filename(self, filename, idx: int):
        """Organize the record filename according to the indexes."""

        return f"{filename}_{idx}.json"

    def save_one_record(
        self, sample: BaseQASample, output: dict, statistic: dict, sample_name: str
    ):
        """Save one record to the disk."""
        with open(
            f"{self.record_dir_path}/{self.get_filename(self.output_filename, sample_name)}",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(output, file)

        with open(
            f"{self.record_dir_path}/{self.get_filename(self.sample_filename, sample_name)}",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(sample, file)

        with open(
            f"{self.record_dir_path}/{self.get_filename(self.statistic_filename, sample_name)}",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(statistic, file)
