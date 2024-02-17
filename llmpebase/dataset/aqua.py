"""
The datasource interface for the Algebra Question Answering with Rationales
(AQUA) dataset.
The detailed information is shown in 
https://huggingface.co/datasets/aqua_rat
"""

import os

import glob
from datasets import load_dataset

from llmpebase.dataset import base
from llmpebase.dataset.data_generic import (
    DatasetMetaCatalog,
    DatasetCatalog,
    BaseQASample,
    BaseQASampleInfo,
    DatasetStatistics,
)


class AQUADataset(base.BaseDataset):
    """
    An interface for the AQUA-RAT dataset.
    """

    def create_data_catalog(self):
        phase_data = load_dataset(
            self.data_meta_catalog["huggingface_dataname"],
            split=self.phase,
            trust_remote_code=True,
        )
        n_items = len(phase_data)
        collected_items = [
            BaseQASampleInfo(
                sample_id=i + 1,
                sample_field="Math",
                sample_problem="Algebra",
                sample_dataset="AQUA-RAT",
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
        sample_info = self.data_catalog.data_samples[idx]

        phase_data = load_dataset(
            self.data_meta_catalog["huggingface_dataname"],
            split=self.phase,
            trust_remote_code=True,
        )
        sample = phase_data[idx]

        question = sample["question"]
        options = sample["options"]
        rationale = sample["rationale"]
        groundtruth = sample["correct"]

        options_str = "\n".join(options)
        question = f"""{question}\nSelect the correct option from the following options.\n{options_str}"""

        answer, conclusion, _ = self.gt_extractor.forward(rationale)

        return BaseQASample(
            question=question,
            answer=answer,
            conclusion=conclusion,
            groundtruth=groundtruth,
            auxiliary={
                "rationale": rationale,
                "sample_info": sample_info,
            },
        )


class DataSource(base.DataSource):
    """The BBH datasource."""

    def __init__(self):
        super().__init__()

        self.base_dataset = AQUADataset

    def download_data(self, phase):
        """Download the data from the huggingface and create
        links to the current folder for visualization."""
        if os.path.exists(self.data_meta_catalog["split_path"][phase]):
            return

        data_name = self.data_meta_catalog["huggingface_dataname"]

        load_dataset(data_name, split=phase, trust_remote_code=True)
        download_path = os.path.join(
            "~/.cache/huggingface/datasets",
            data_name,
        )
        # Search the download path to get the phase data
        download_path = os.path.expanduser(download_path)
        pattern = os.path.join(download_path, "**", f"*{phase}*.arrow")
        # Find all files in directory and subdirectories that match the pattern
        file = glob.glob(pattern, recursive=True)[0]

        # Create the link to the current data folder
        if not os.path.exists(self.data_meta_catalog["split_path"][phase]):
            os.symlink(file, self.data_meta_catalog["split_path"][phase])

    def create_meta_catalog(self):
        """Configure the dataset."""
        return DatasetMetaCatalog(
            dataset_name="AQUA-RAT",
            task_type="Mathematical Reasoning",
            dataset_path=self.data_path,
            split_path={
                "train": os.path.join(self.data_path, "train.arrow"),
                "test": os.path.join(self.data_path, "test.arrow"),
                "validation": os.path.join(self.data_path, "validation.arrow"),
            },
            huggingface_dataname="aqua_rat",
        )
