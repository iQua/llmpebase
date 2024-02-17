"""
The datasource interface for the commonsenseQA (CSQA) dataset.
The detailed information is shown in 
https://huggingface.co/datasets/commonsense_qa
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


class CSQADataset(base.BaseDataset):
    """
    An interface for the CSQA dataset.
    """

    def create_data_catalog(self):
        phase_data = load_dataset(
            self.data_meta_catalog["huggingface_dataname"],
            split=self.phase,
            trust_remote_code=True,
        )

        problem_category = []
        collect_items = []
        category_samples = defaultdict(list)
        category_info = defaultdict(dict)
        for example in phase_data:
            concept = tools.format_term(example["question_concept"])
            if concept not in problem_category:
                problem_category.append(concept)

            collect_items.append(
                BaseQASampleInfo(
                    sample_id=example["id"],
                    sample_problem=concept,
                    sample_filepath=self.phase_data_path,
                )
            )
            category_samples[concept].append(len(collect_items) - 1)
            category_info[concept]["num_samples"] = len(category_samples[concept])

        return DatasetCatalog(
            data_phase=self.phase,
            data_samples=collect_items,
            category_samples=category_samples,
            problem_categories=problem_category,
            data_statistics=DatasetStatistics(
                num_samples=len(collect_items), category_info=category_info
            ),
        )

    def get_sample(self, idx):
        phase_data = load_dataset(
            self.data_meta_catalog["huggingface_dataname"],
            split=self.phase,
            trust_remote_code=True,
        )
        # 'id', 'question', 'question_concept', 'choices', 'answerKey'
        phase_sample = phase_data[idx]

        sample_info = self.data_catalog.data_samples[idx]
        sample_problem = sample_info["sample_problem"]

        question = phase_sample["question"]
        options = phase_sample["choices"]["text"]
        choice_letters = phase_sample["choices"]["label"]
        option_contents = dict(zip(choice_letters, options))
        options_str = [f"({opt}) {content}" for opt, content in option_contents.items()]
        options_str = "\n".join(options_str)

        question = f"""{question}\nSelect one correct option from the following options.\n{options_str}"""
        target_option = phase_sample["answerKey"]

        target_answer = " "
        if len(target_option.strip()) != 0:
            target_answer = option_contents[target_option]

        return BaseQASample(
            question=question,
            answer=f"{target_option}, {target_answer}",
            conclusion=f"{target_option}, {target_answer}",
            groundtruth=target_option,
            auxiliary={
                "options": options,
                "choice_letters": choice_letters,
                "option_str": options_str,
                "sample_problem": sample_problem,
            },
        )


class DataSource(base.DataSource):
    """The CSQA datasource."""

    def __init__(self):
        super().__init__()

        self.base_dataset = CSQADataset

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
            dataset_name="CommonsenseQA",
            task_type="Commonsense Reasoning",
            dataset_path=self.data_path,
            split_path={
                "train": os.path.join(self.data_path, "train.arrow"),
                "test": os.path.join(self.data_path, "test.arrow"),
                "validation": os.path.join(self.data_path, "validation.arrow"),
            },
            huggingface_dataname="commonsense_qa",
        )

    def get_test_set(self):
        """Obtains the validation dataset."""
        phase = "validation"
        return self.get_phase_dataset(phase)
