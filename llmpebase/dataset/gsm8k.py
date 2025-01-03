""" 
The datasource inference for the GSM8K dataset.
The detailed information of it is shown in 
https://huggingface.co/datasets/gsm8k
"""

import os

import pandas as pd


from llmpebase.dataset import base
from llmpebase.dataset.data_generic import (
    DatasetMetaCatalog,
    DatasetCatalog,
    BaseQASample,
    BaseQASampleInfo,
    DatasetStatistics,
)


class GSM8KDataset(base.BaseDataset):
    """
    An interface for the GSM8K dataset.
    """

    def create_data_catalog(self):
        data_frame = pd.read_parquet(self.phase_data_path, engine="pyarrow")
        n_items = data_frame.shape[0]

        collected_items = [
            BaseQASampleInfo(
                sample_id=i + 1,
                sample_field="Math",
                sample_problem="Algebra",
                sample_dataset="GSM8K",
                sample_filepath=self.phase_data_path,
            )
            for i in range(n_items)
        ]

        return DatasetCatalog(
            data_phase=self.phase,
            problem_fields=["Math"],
            problem_categories={"Math": ["Algebra"]},
            category_samples={"Math": {"Algebra": list(range(n_items))}},
            data_samples=collected_items,
            data_statistics=DatasetStatistics(
                num_samples=n_items,
                category_info={"Math": {"Algebra": {"num_samples": n_items}}},
            ),
        )

    def get_sample(self, idx):
        """Get one sample."""
        sample_info = self.data_catalog.data_samples[idx]
        sample_path = sample_info["sample_filepath"]
        data_frame = pd.read_parquet(sample_path, engine="pyarrow")

        raw_answer = data_frame.iloc[idx, -1]
        # Extract the answer, conclusion, and groundtruth from the raw answer
        answer, conclusion, groundtruth = self.gt_extractor.forward(raw_answer)

        return BaseQASample(
            question=data_frame.iloc[idx, 0],
            answer=answer,
            conclusion=conclusion,
            groundtruth=groundtruth,
            auxiliary={"raw_answer": raw_answer, "sample_info": sample_info},
        )


class DataSource(base.DataSource):
    """The GSM8K datasource."""

    def __init__(self):
        super().__init__()

        self.base_dataset = GSM8KDataset

    def create_meta_catalog(self):
        """Configure the dataset."""
        return DatasetMetaCatalog(
            dataset_name="GSM8K",
            task_type="Mathematical Reasoning",
            dataset_path=self.data_path,
            split_path={
                "train": os.path.join(self.data_path, "train.parquet"),
                "test": os.path.join(self.data_path, "test.parquet"),
                "validation": os.path.join(self.data_path, "test.parquet"),
            },
        )
