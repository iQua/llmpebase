"""
An interface of datasets
"""
import logging

from llmpebase.dataset.gsm8k import DataSource as gsm8k_datasource
from llmpebase.dataset.mmlu import DataSource as mmlu_datasource
from llmpebase.dataset.game24 import DataSource as game24_datasource
from llmpebase.dataset.math import DataSource as math_datasource
from llmpebase.dataset.bbh import DataSource as bbh_datasource
from llmpebase.dataset.theoremqa import DataSource as theoremqa_datasource
from llmpebase.dataset.csqa import DataSource as csqa_datasource

datasources = {
    "GSM8K": gsm8k_datasource,
    "MMLU": mmlu_datasource,
    "GameOf24": game24_datasource,
    "MATH": math_datasource,
    "BBH": bbh_datasource,
    "TheoremQA": theoremqa_datasource,
    "CSQA": csqa_datasource,
}


def define_dataset(data_config):
    """Define the datasets based on the config file."""
    data_name = data_config["data_name"]
    logging.info("Define the dataset %s", data_name)
    return datasources[data_name]()
