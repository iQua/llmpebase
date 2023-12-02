""" 
The datasource inferance for the TheoremQA (TQA) dataset.
The detaild information of it is shown in 
https://github.com/wenhuchen/TheoremQA
"""
import os
import json
from collections import defaultdict

from llmpebase.datasets import base
from llmpebase.datasets.data_generic import (
    DatasetMetaCatalog,
    DatasetCatalog,
    BaseQASample,
    BaseQASampleInfo,
    DatasetStatistics,
)
from llmpebase.utils import extractor


def format_theorem(name: str):
    """Convert a theorem to the format one."""

    return name.replace("_", " ").title()


class TheoremQADataset(base.BaseDataset):
    """
    An interface for the TheoremQA dataset.
    """

    def create_data_catalog(self):
        with open(self.phase_data_path, "r", encoding="utf-8") as f:
            # Load examples
            data = json.load(f)
        collect_items = []
        problem_category = defaultdict(list)
        category_info = defaultdict(dict)
        category_samples = defaultdict(dict)
        for idx, example in enumerate(data):
            field = example["field"]
            subfield = example["subfield"]

            # Get the explanation path
            root_path = os.path.dirname(self.phase_data_path)
            explain_path = (
                None
                if "/" not in example["explanation"]
                else os.path.join(root_path, example["explanation"])
            )
            # Get the image path
            image_path = (
                None
                if example["Picture"] is None
                else os.path.join(root_path, example["Picture"])
            )

            collect_items.append(
                BaseQASampleInfo(
                    sample_id=idx,
                    sample_problem=field,
                    sample_filepath=self.phase_data_path,
                    auxiliary={
                        # The theorem used in the answer
                        "theorem": format_theorem(example["theorem"]),
                        # The detailed reasoning steps
                        "explain_path": explain_path,
                        # Visual path
                        "visual_path": image_path,
                        "problem_subfield": subfield,
                    },
                )
            )
            if subfield not in problem_category[field]:
                problem_category[field].append(subfield)

            if subfield not in category_info[field]:
                category_info[field][subfield] = {}
                category_info[field][subfield]["num_samples"] = 0
                category_samples[field][subfield] = []
                category_samples[field][subfield].append(idx)
            else:
                category_info[field][subfield]["num_samples"] += 1
                category_samples[field][subfield].append(idx)

        return DatasetCatalog(
            data_phase=self.phase,
            data_samples=collect_items,
            category_samples=category_samples,
            problem_category=problem_category,
            data_statistics=DatasetStatistics(
                num_samples=len(collect_items), category_info=category_info
            ),
        )

    def get_sample(self, idx: int):
        sample_info = self.data_catalog.data_samples[idx]
        sample_filepath = sample_info["sample_filepath"]
        with open(sample_filepath, "r", encoding="utf-8") as f:
            sample = json.load(f)[idx]

        explanation = sample["Answer"]
        conclusion = sample["Answer"]
        explain_path = sample_info["auxiliary"]["explain_path"]
        if explain_path is not None:
            with open(explain_path, "r", encoding="utf-8") as f:
                explanation = f.read(f)
            conclusion = extractor.extract_sentences(explanation)[-1]

        return BaseQASample(
            question=sample["Question"],
            answer=explanation,
            conclusion=conclusion,
            groundtruth=sample["Answer"],
            auxiliary={
                "answer_type": sample["Answer_type"],
                "sample_problem": sample_info["sample_problem"],
                "problem_subfield": sample_info["auxiliary"]["problem_subfield"],
            },
        )

    def get_problem_sample_indexs(self, problem_name):
        """Get sample indexs of one problem."""
        # Get the problem, i.e., the subfield, belong
        # to which field
        field = [
            category
            for category in self.data_catalog.problem_category
            if problem_name in self.data_catalog.problem_category[category]
        ][0]

        return self.data_catalog.category_samples[field][problem_name]


class DataSource(base.DataSource):
    """The TheoremQA datasource."""

    def __init__(self):
        super().__init__()

        self.base_dataset = TheoremQADataset

    def create_meta_catalog(self):
        """Configure the dataset."""
        return DatasetMetaCatalog(
            dataset_name="TheoremQA",
            task_type="Question Answering for Math, EE & CS, Physics and Finance",
            dataset_path=self.data_path,
            split_path={
                "train": os.path.join(
                    self.data_path, "TheoremQA-main", "theoremqa_test.json"
                ),
                "test": os.path.join(
                    self.data_path, "TheoremQA-main", "theoremqa_test.json"
                ),
                "visual_test": os.path.join(
                    self.data_path,
                    "TheoremQA-main",
                    "theoremqa_visual_subset_test.json",
                ),
                "val": os.path.join(
                    self.data_path, "TheoremQA-main", "theoremqa_test.json"
                ),
                "images": os.path.join(self.data_path, "TheoremQA-main", "images"),
            },
        )
