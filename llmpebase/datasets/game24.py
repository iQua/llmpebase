""" 
The datasource inferance for the Game of 24 dataset.
"""
import os

import pandas as pd

from llmpebase.datasets import base
from llmpebase.datasets.data_generic import (
    DatasetMetaCatalog,
    DatasetCatalog,
    BaseQASample,
    BaseQASampleInfo,
    DatasetStatistics,
)


class GameOf24Dataset(base.BaseDataset):
    """
    An interface for the GameOf24 dataset.
    """

    def create_data_catalog(self):
        data_frame = pd.read_csv(self.phase_data_path)
        n_itmes = data_frame.shape[0]

        collected_items = [
            BaseQASampleInfo(
                sample_id=data_frame["Rank"].iloc[i].item(),
                sample_task="Algebra",
                sample_filepath=self.phase_data_path,
            )
            for i in range(n_itmes)
        ]
        return DatasetCatalog(
            data_phase=self.phase,
            qa_sample_files=collected_items,
            data_statistics=DatasetStatistics(num_samples=n_itmes),
        )

    def get_sample(self, idx):
        """Get one sample."""
        sample_path = self.data_catalog.qa_sample_files[idx]["sample_filepath"]
        sample_task = self.data_catalog.qa_sample_files[idx]["sample_task"]
        data_frame = pd.read_csv(sample_path)
        return BaseQASample(
            question=data_frame["Puzzles"].iloc[idx],
            answer="",
            conclusion="",
            groundtruth=24,
            auxiliary={
                "solved_rate": data_frame["Solved rate"].iloc[idx],
                "AMT": data_frame["AMT (s)"].iloc[idx],
                "1_sigma_Mean": data_frame["1-sigma Mean (s)"].iloc[idx],
                "1_sigma_STD": data_frame["1-sigma STD (s)"].iloc[idx],
                "sample_task": sample_task,
            },
        )


class DataSource(base.DataSource):
    """The GameOf24 dataset."""

    def __init__(self):
        super().__init__()

        self.base_dataset = GameOf24Dataset

    def create_meta_catalog(self):
        """Configure the dataset."""
        return DatasetMetaCatalog(
            dataset_name="GameOf24",
            problem_type="Mathematical Reasoning",
            dataset_path=self.data_path,
            split_path={
                "train": os.path.join(self.data_path, "24.csv"),
                "test": os.path.join(self.data_path, "24.csv"),
                "val": os.path.join(self.data_path, "24.csv"),
            },
        )
