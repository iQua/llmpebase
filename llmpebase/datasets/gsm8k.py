""" 
The datasource inferance for the GSM8K dataset.
The detaild information of it is shown in 
https://huggingface.co/datasets/gsm8k
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
from llmpebase.utils import extractor


class GSM8KDataset(base.BaseDataset):
    """
    An interface for the GSM8K dataset.
    """

    def create_data_catalog(self):
        data_frame = pd.read_parquet(self.phase_data_path, engine="pyarrow")
        n_itmes = data_frame.shape[0]

        collected_items = [
            BaseQASampleInfo(
                sample_id=i + 1,
                sample_problem="Algebra",
                sample_filepath=self.phase_data_path,
            )
            for i in range(n_itmes)
        ]

        return DatasetCatalog(
            data_phase=self.phase,
            data_samples=collected_items,
            category_samples={"Algebra": list(range(n_itmes))},
            data_statistics=DatasetStatistics(
                num_samples=n_itmes, category_info={"Algebra": n_itmes}
            ),
            problem_category=["Algebra"],
        )

    def get_sample(self, idx):
        """Get one sample."""
        sample_path = self.data_catalog.data_samples[idx]["sample_filepath"]
        sample_problem = self.data_catalog.data_samples[idx]["sample_problem"]
        data_frame = pd.read_parquet(sample_path, engine="pyarrow")

        raw_answer = data_frame.iloc[idx, -1]
        answer = raw_answer.split("####")[0]
        answer = answer.replace("\n", " ").rstrip()
        target_sent = extractor.extract_sentences(answer)[-1]
        groundtruth = extractor.extract_target_result(target_sent, target_format="#")
        groundtruth = groundtruth[-1]

        return BaseQASample(
            question=data_frame.iloc[idx, 0],
            answer=answer,
            conclusion=target_sent,
            groundtruth=groundtruth,
            auxiliary={"raw_answer": raw_answer, "sample_problem": sample_problem},
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
                "val": os.path.join(self.data_path, "test.parquet"),
            },
        )
