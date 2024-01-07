"""
Basic components used in the dataset.
"""
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

from transformers.utils import ModelOutput as FieldFrozenContainer

# from vgbase.utils.generic_components import FieldFrozenContainer


@dataclass
class DatasetMetaCatalog(FieldFrozenContainer):
    """The meta data information of one dataset.

    The design principle of this class derives from the
    `detectron2` package.

    It holds the meta information of one dataset

    Args:
        dataset_name: Holding the name of the dataset.
        task_type: Holding the name of the problem of this dataset.
        dataset_path: Holding the root path of the dataset.
        splits: Holding the split information of the dataset,
            the key is the split name, and the value is the
            path of the split.
    """

    dataset_name: str
    task_type: Optional[str] = None
    dataset_path: Optional[str] = None
    split_path: Optional[Dict[str, str]] = None

    huggingface_dataname: Optional[str] = None


@dataclass
class BaseQASampleInfo(FieldFrozenContainer):
    """The basic common items contained in one question-answer sample.

    Args:
        sample_id: The id of the sample.
        sample_problem: The name of the problem.
        qa_filepath: The filepath that storess the question and the answer.
    """

    sample_id: Optional[str] = None
    sample_problem: Optional[str] = None
    sample_filepath: Optional[str] = None

    auxiliary: Optional[Dict] = None


@dataclass
class DatasetStatistics(FieldFrozenContainer):
    """The statistics of the dataset."""

    num_samples: int
    category_info: Optional[Dict[str, Any]] = None


@dataclass
class DatasetCatalog(FieldFrozenContainer):
    """The samples catalog of one dataset.

    Args:
        data_phase: A str showing the phase of the dataset.
        problem_category: A list containing the information of
         the problem category.
        data_samples: A list containing the information of
         samples.
        category_samples: A dict containing the sample positions
         of each category.
        data_statistics: A DatasetStatistics holding the information
         the statistics of the dataset.

    """

    data_phase: str
    problem_category: List[Union[str, dict]] = None
    data_samples: Optional[List[BaseQASampleInfo]] = None
    category_samples: Optional[Dict[str, List[int]]] = None
    data_statistics: Optional[DatasetStatistics] = None


@dataclass
class BaseQASample(FieldFrozenContainer):
    """The basic common items contained in one question-answer sample.

    Args:
        question: The given question required to be addressed
        answer: The obtained answer of the given question.
        conclusion: The conclusion, which is generally the summary,
         of the given answer.
        groundtruth: The groundtruth, which is generally the single numbers,
         of the solution of given question.

        auxiliary: The auxiliary information of the sample.
    """

    question: Optional[str] = None
    answer: Optional[str] = None
    conclusion: Optional[str] = None
    groundtruth: Optional[str] = None

    auxiliary: Optional[Dict] = None


@dataclass
class MATHDatasetStatistics(DatasetStatistics):
    """The statistics of the dataset.

    Args:
        category_info: A dict that holds the information of the category,
         i.e., category_name:
            - num_samples: The number of samples of the category.
            - level id: n_samples
        difficulty_count: A dict that holds the information of the difficulty,
         i.e., level: n_samples
    """

    difficulty_count: Optional[Dict[str, int]] = None


@dataclass
class MATHDatasetCatalog(DatasetCatalog):
    """The samples catalog of one dataset."""

    difficulty_level: List[str] = None
    category_difficulty: Optional[Dict[str, List[str]]] = None
