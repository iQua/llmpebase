"""
The datasource interface for the Simple Variations on Arithmetic Math word Problems
(SVAMP) dataset.
The detailed information is shown in 
https://huggingface.co/datasets/ChilleD/SVAMP
"""

import os
from collections import defaultdict

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
from llmpebase.utils import tools


class SVAMPDataset(base.BaseDataset):
    """
    An interface for the SVAMP dataset.
    """

    def create_data_catalog(self):
        phase_data = load_dataset(
            self.data_meta_catalog["huggingface_dataname"],
            split=self.phase,
            trust_remote_code=True,
        )
        # Body, Type, Equation, Question, ID, Answer,
        problem_sub_category = []
        collect_items = []
        sub_category_samples = defaultdict(list)
        sub_category_info = defaultdict(dict)
        for example in phase_data:
            problem_type = tools.format_term(example["Type"])
            if problem_type not in problem_sub_category:
                problem_sub_category.append(problem_type)

            collect_items.append(
                BaseQASampleInfo(
                    sample_id=example["ID"],
                    sample_field="Math",
                    sample_problem=f"Algebra/{problem_type}",
                    sample_dataset="SVAMP",
                    sample_filepath=self.phase_data_path,
                )
            )
            sub_category_samples[problem_type].append(len(collect_items) - 1)
            sub_category_info[problem_type]["num_samples"] = len(
                sub_category_samples[problem_type]
            )

        return DatasetCatalog(
            data_phase=self.phase,
            problem_fields=["Math"],
            problem_categories={"Math": {"Algebra": problem_sub_category}},
            category_samples=sub_category_samples,
            data_samples=collect_items,
            data_statistics=DatasetStatistics(
                num_samples=len(collect_items),
                category_info={"Math": {"Algebra": sub_category_info}},
            ),
        )

    def get_sample(self, idx):
        phase_data = load_dataset(
            self.data_meta_catalog["huggingface_dataname"],
            split=self.phase,
            trust_remote_code=True,
        )
        # Body, Type, Equation, ID, Question, Answer,
        phase_sample = phase_data[idx]

        sample_info = self.data_catalog.data_samples[idx]

        description = phase_sample["Body"]
        target = phase_sample["Question"]
        conclusion = phase_sample["Equation"]
        answer = phase_sample["Answer"]

        return BaseQASample(
            question=f"{description} {target}",
            answer=f"{conclusion}={answer}",
            conclusion=conclusion,
            groundtruth=answer,
            auxiliary={
                "sample_info": sample_info,
            },
        )


class DataSource(base.DataSource):
    """The BBH datasource."""

    def __init__(self):
        super().__init__()

        self.base_dataset = SVAMPDataset

    def download_data(self, phase):
        """Download the data from the huggingface and create
        links to the current folder for visualization."""
        if os.path.exists(self.data_meta_catalog["split_path"][phase]):
            return

        hug_cache_path = os.path.expanduser("~/.cache/huggingface/datasets")

        data_name = self.data_meta_catalog["dataset_name"]
        hug_data_name = self.data_meta_catalog["huggingface_dataname"]

        load_dataset(hug_data_name, split=phase, trust_remote_code=True)

        # Search the root folder to get the dataset folder
        download_path = os.path.join(
            hug_cache_path,
            hug_data_name,
        )
        # If the direct name is not existed, which is generally caused by
        # the huggingface will rename the dataset folder to from "Author/Name"
        # to "Author__Name", such as ChilleD/SVAMP -> ChilleD__SVAMP
        # search the dataname
        if not os.path.exists(download_path):
            data_folder = [
                folder
                for folder in os.scandir(hug_cache_path)
                if folder.is_dir() and data_name.lower() in folder.name.lower()
            ]
        if len(data_folder):
            download_path = data_folder[0].path
        else:
            raise ValueError(
                f"Cannot find the dataset folder for {data_name} in {hug_cache_path}"
            )

        # Search the download path to get the phase data
        pattern = os.path.join(download_path, "**", f"*{phase}*.arrow")
        # Find all files in directory and subdirectories that match the pattern
        file = glob.glob(pattern, recursive=True)[0]

        # Create the link to the current data folder
        if not os.path.exists(self.data_meta_catalog["split_path"][phase]):
            os.symlink(file, self.data_meta_catalog["split_path"][phase])

    def create_meta_catalog(self):
        """Configure the dataset."""
        return DatasetMetaCatalog(
            dataset_name="SVAMP",
            task_type="Mathematical Reasoning",
            dataset_path=self.data_path,
            split_path={
                "train": os.path.join(self.data_path, "train.arrow"),
                "test": os.path.join(self.data_path, "test.arrow"),
            },
            huggingface_dataname="ChilleD/SVAMP",
        )
