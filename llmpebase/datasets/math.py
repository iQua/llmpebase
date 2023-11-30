""" 
The datasource inferance for the MATH dataset.
The detaild information of it is shown in 
https://people.eecs.berkeley.edu/~hendrycks/MATH.tar
"""
import os
import json
import glob
from collections import defaultdict

from llmpebase.datasets import base
from llmpebase.datasets.data_generic import (
    DatasetMetaCatalog,
    MATHDatasetCatalog,
    BaseQASample,
    BaseQASampleInfo,
    MATHDatasetStatistics,
)

from llmpebase.utils import extractor


class AddableDict(dict):
    """A dict to merge two dicts by adding the values of the same key."""

    def update(self, other):
        for key, value in other.items():
            if key in self:
                self[key] += value
            else:
                self[key] = value


def count_category(category_path: str) -> tuple:
    """Count the data information in the category."""
    category_levels = defaultdict(int)
    collect_items = []
    qa_files = glob.glob(f"{category_path}/*.json")
    category_name = ""
    for filepath in qa_files:
        file_id = os.path.basename(filepath).split(".json")[0]

        # Load the data file
        with open(filepath, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            category_name = data["type"]
            level = data["level"]

        collect_items.append(
            BaseQASampleInfo(
                sample_id=file_id,
                sample_task=category_name,
                sample_filepath=filepath,
            )
        )

        category_levels[level] += 1

    return category_name, category_levels, collect_items


class MATHDataset(base.BaseDataset):
    """
    An interface for the MATH dataset.
    """

    def __init__(self, data_meta_catalog: DatasetMetaCatalog, phase: str):
        super().__init__(data_meta_catalog, phase)

        self.customize_data_catalog = MATHDatasetCatalog

    def create_data_catalog(self):
        # Collect all category folders of the MATH dataset
        folders = [
            folder
            for folder in glob.glob(os.path.join(self.phase_data_path, "*"))
            if os.path.isdir(folder)
        ]
        # Visit each category folder to get data information
        category_info = defaultdict(dict)
        difficulty_count = AddableDict()
        collect_items = []
        for category_path in folders:
            # Get the info of the category
            category_name, category_levels, items = count_category(category_path)
            # Update the category info to the category_info
            category_info[category_name]["num_samples"] = len(items)
            category_info[category_name].update(category_levels)
            difficulty_count.update(category_levels)
            collect_items.extend(items)

        return MATHDatasetCatalog(
            data_phase=self.phase,
            problem_category=list(category_info.keys()),
            data_statistics=MATHDatasetStatistics(
                num_samples=len(collect_items),
                category_info=category_info,
                difficulty_count=difficulty_count,
            ),
            qa_sample_files=collect_items,
        )

    def get_sample(self, idx: int):
        """Get one sample from the file."""
        sample_info = self.data_catalog.qa_sample_files[idx]
        sample_task = self.data_catalog.qa_sample_files[idx]["sample_task"]

        sample_filepath = sample_info["sample_filepath"]

        with open(sample_filepath, "r", encoding="utf-8") as json_file:
            # Load the JSON data from the file
            data = json.load(json_file)

        solution = data["solution"].rstrip()

        sents = extractor.extract_sentences(solution)
        sents = [sent for sent in sents if "=" in sent or "\\boxed" in sent]
        conclusion = sents[-1]

        groundtruths = extractor.extract_target_equations(conclusion)
        result = groundtruths[-1]
        final_result = extractor.extract_equation_result(result)

        return BaseQASample(
            question=data["problem"],
            answer=solution,
            conclusion=conclusion,
            groundtruth=final_result,
            auxiliary={
                "level": data["level"],
                "category": data["type"],
                "sample_task": sample_task,
            },
        )


class DataSource(base.DataSource):
    """The MATH datasource."""

    def __init__(self):
        super().__init__()

        self.base_dataset = MATHDataset

    def create_meta_catalog(self):
        """Configure the dataset."""
        return DatasetMetaCatalog(
            dataset_name="MATH",
            problem_type="Mathematical Reasoning",
            dataset_path=self.data_path,
            split_path={
                "train": os.path.join(self.data_path, "MATH/train"),
                "test": os.path.join(self.data_path, "MATH/test"),
                "val": os.path.join(self.data_path, "MATH/test"),
            },
        )
