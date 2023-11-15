""" 
The datasource inferance for the Game of 24 dataset.
"""
import os
from typing import Tuple

import torch
import pandas as pd

from llmpebase.datasets import base


class GameOf24Dataset(torch.utils.data.Dataset):
    """
    An interface for the GameOf24 dataset.
    """

    def __init__(self, splits_info, phase):
        # a path showing where the data is stored
        self.splits_info = splits_info
        self.phase = phase

        data_folder = self.splits_info[phase]["path"]
        data_file = self.splits_info[phase]["filename"]
        self.data_path = os.path.join(data_folder, data_file)
        self.data_qas = self.collect_qas()

    def collect_qas(self):
        """Collecting the question and the correspondin answer."""
        data_frame = pd.read_csv(self.data_path)
        n_itmes = data_frame.shape[0]
        collected_items = []
        for row_idx in range(n_itmes):
            collected_items.append(
                {
                    "question": data_frame["Puzzles"].iloc[row_idx],
                    "answer": None,
                    "target_answer": 24,
                    "solved_rate": data_frame["Solved rate"].iloc[row_idx],
                    "AMT": data_frame["AMT (s)"].iloc[row_idx],
                    "1_sigma_Mean": data_frame["1-sigma Mean (s)"].iloc[row_idx],
                    "1_sigma_STD": data_frame["1-sigma STD (s)"].iloc[row_idx],
                }
            )

        return collected_items

    def get_qas(self):
        """Getting the qas of the tasks."""
        return self.data_qas

    def __getitem__(self, idx: Tuple):
        """Get the sample for either training or testing given index."""
        return self.data_qas[idx]

    def __len__(self):
        """obtain the number of samples."""
        return len(self.data_qas)


class DataSource(base.DataSource):
    """The GameOf24 dataset."""

    def __init__(self):
        # Extract the data information from the config file
        super().__init__()

        self.splits_info = {
            "train": {"path": self.data_path, "filename": "24.csv"},
            "test": {"path": self.data_path, "filename": "24.csv"},
        }

    def get_phase_dataset(self, phase: str):
        """Obtain the dataset for the specific phase."""
        self.prepare_source_data(phase)
        # obtain the datacatalog for desired phase
        dataset = GameOf24Dataset(splits_info=self.splits_info, phase=phase)
        return dataset
