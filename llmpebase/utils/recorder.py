"""
Implementation of recorders for saving outputs to files.
"""

import os
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

    def save_one_record(
        self, sample: BaseQASample, output: dict, statistic: dict, sample_idx: int
    ):
        """Save one record to the disk."""
        with open(
            f"{self.record_dir_path}/{self.output_filename}_{sample_idx}.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(output, file)

        with open(
            f"{self.record_dir_path}/{self.sample_filename}_{sample_idx}.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(sample, file)

        with open(
            f"{self.record_dir_path}/{self.statistic_filename}_{sample_idx}.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(statistic, file)
