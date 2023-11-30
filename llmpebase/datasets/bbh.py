""" 
The datasource inferance for the BIG-Bench Hard (BBH) dataset.
The detaild information of it is shown in 
https://github.com/suzgunmirac/BIG-Bench-Hard
"""
import os
import json
import glob
from collections import defaultdict

from llmpebase.datasets import base
from llmpebase.datasets.data_generic import (
    DatasetMetaCatalog,
    DatasetCatalog,
    BaseQASample,
    BaseQASampleInfo,
    DatasetStatistics,
)


def extract_problem_name(filename: str):
    """
    Extract the problem name from the filepath by
    removing the extension,
    useless characters and
    capitalizing the first letter of each word.
    """
    filename = filename.split(".json")[0]
    return filename.replace("_", " ").rstrip().title()


class BBHDataset(base.BaseDataset):
    """
    An interface for the BBH dataset.
    """

    def create_data_catalog(self):
        json_files = glob.glob(self.phase_data_path + "/*.json")

        problem_category = []
        collect_items = []
        category_info = defaultdict(dict)
        for filepath in json_files:
            with open(filepath, "r", encoding="utf-8") as f:
                # Load the examples as the list
                examples = json.load(f)["examples"]
            num_examples = len(examples)
            category_name = extract_problem_name(os.path.basename(filepath))
            problem_category.append(category_name)
            collect_items.extend(
                [
                    BaseQASampleInfo(
                        sample_id=f"{category_name}_{idx}",
                        sample_problem=category_name,
                        sample_filepath=filepath,
                    )
                    for idx in range(num_examples)
                ]
            )

            category_info[category_name]["num_samples"] = num_examples

        return DatasetCatalog(
            data_phase=self.phase,
            qa_sample_info=collect_items,
            problem_category=problem_category,
            data_statistics=DatasetStatistics(
                num_samples=len(collect_items), category_info=category_info
            ),
        )

    def get_sample(self, idx: int):
        """Get one sample from the file."""
        sample_info = self.data_catalog.qa_sample_info[idx]
        sample_id = sample_info["sample_id"]
        sample_problem = sample_info["sample_problem"]

        sample_filepath = sample_info["sample_filepath"]

        with open(sample_filepath, "r", encoding="utf-8") as json_file:
            examples = json.load(json_file)["examples"]
        sample_idx = sample_id.split("_")[-1]

        sample_data = examples[sample_idx]
        return BaseQASample(
            question=sample_data["input"],
            answer=sample_data["target"],
            conclusion=sample_data["target"],
            groundtruth=sample_data["target"],
            auxiliary={"sample_problem": sample_problem, "sample_idx": sample_idx},
        )


class DataSource(base.DataSource):
    """The BBH datasource."""

    def __init__(self):
        super().__init__()

        self.base_dataset = BBHDataset

    def create_meta_catalog(self):
        """Configure the dataset."""
        return DatasetMetaCatalog(
            dataset_name="BBH",
            task_type="Symbolic & Text Reasoning",
            dataset_path=self.data_path,
            split_path={
                "train": os.path.join(self.data_path, "BIG-Bench-Hard-main", "bbh"),
                "test": os.path.join(self.data_path, "BIG-Bench-Hard-main", "bbh"),
                "val": os.path.join(self.data_path, "BIG-Bench-Hard-main", "bbh"),
            },
        )
