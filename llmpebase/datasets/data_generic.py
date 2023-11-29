"""
Basic components used in the dataset.
"""
from typing import List, Optional, Dict
from dataclasses import dataclass


from vgbase.utils.generic_components import FieldFrozenContainer


@dataclass
class DatasetStatistics(FieldFrozenContainer):
    """The statistics of the dataset."""

    num_samples: int


@dataclass
class BaseQASampleInfo(FieldFrozenContainer):
    """The basic common items contained in one question-answer sample.

    Args:
        sample_id: The id of the sample.
        sample_task: The name of the task.
        qa_filepath: The filepath that storess the question and the answer.
    """

    sample_id: Optional[str] = None
    sample_task: Optional[str] = None
    sample_filepath: Optional[str] = None

    auxiliary: Optional[Dict] = None


@dataclass
class DatasetMetaCatalog(FieldFrozenContainer):
    """The meta data information of one dataset.

    The design principle of this class derives from the
    `detectron2` package.

    It holds the meta information of one dataset

    Args:
        dataset_name: Holding the name of the dataset.
        problem_type: Holding the name of the problem of this dataset.
        dataset_path: Holding the root path of the dataset.
        splits: Holding the split information of the dataset,
            the key is the split name, and the value is the
            path of the split.
    """

    dataset_name: str
    problem_type: Optional[str] = None
    dataset_path: Optional[str] = None
    split_path: Optional[Dict[str, str]] = None


@dataclass
class DatasetCatalog(FieldFrozenContainer):
    """The samples catalog of one dataset."""

    data_phase: str
    data_statistics: Optional[DatasetStatistics] = None
    qa_sample_files: Optional[List[BaseQASampleInfo]] = None


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
class MMLUDatasetStatistics(DatasetStatistics):
    """The statistics of the dataset."""

    category_count: Optional[Dict[str, int]] = None


@dataclass
class MMLUDatasetCatalog(DatasetCatalog):
    """The samples catalog of one dataset."""

    problem_category: List[str] = None


@dataclass
class MATHDatasetStatistics(DatasetStatistics):
    """The statistics of the dataset."""

    category_count: Optional[Dict[str, int]] = None
    category_difficulty_count: Optional[Dict[str, Dict[str, int]]] = None
    difficulty_count: Optional[Dict[str, int]] = None


@dataclass
class MATHDatasetCatalog(DatasetCatalog):
    """The samples catalog of one dataset."""

    problem_category: List[str] = None
    difficulty_level: List[str] = None
    category_difficulty: Optional[Dict[str, List[str]]] = None
