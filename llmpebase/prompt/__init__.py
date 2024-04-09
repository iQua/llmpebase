"""
All prompts used in the LlmpeBase project.
"""

import logging


from llmpebase.prompt.system_prompt import BaseSystemPrompts, GameOf24SystemPrompts

from llmpebase.prompt.thought_prompt import BaseThoughtPrompts

from llmpebase.prompt.chain_prompt import (
    ChainOutcomeCommentPrompts,
    Gameof24ChainOutcomeCommentPrompts,
)


system_prompts = {
    "mmlu": BaseSystemPrompts,
    "gsm8k": BaseSystemPrompts,
    "gameof24": GameOf24SystemPrompts,
    "math": BaseSystemPrompts,
    "bbh": BaseSystemPrompts,
    "theoremqa": BaseSystemPrompts,
    "csqa": BaseSystemPrompts,
    "aqua": BaseSystemPrompts,
    "svamp": BaseSystemPrompts,
}


# Define different types of thought prompters for different datasets
thought_prompts = {
    "mmlu": BaseThoughtPrompts,
    "gsm8k": BaseThoughtPrompts,
    "gameof24": BaseThoughtPrompts,
    "math": BaseThoughtPrompts,
    "bbh": BaseThoughtPrompts,
    "theoremqa": BaseThoughtPrompts,
    "csqa": BaseThoughtPrompts,
    "aqua": BaseThoughtPrompts,
    "svamp": BaseThoughtPrompts,
}

chain_comment_prompts = {
    "mmlu": {
        "outcome": ChainOutcomeCommentPrompts,
    },
    "gsm8k": {
        "outcome": ChainOutcomeCommentPrompts,
    },
    "gameof24": {
        "outcome": Gameof24ChainOutcomeCommentPrompts,
    },
    "math": {
        "outcome": ChainOutcomeCommentPrompts,
    },
    "bbh": {
        "outcome": ChainOutcomeCommentPrompts,
    },
    "theoremqa": {
        "outcome": ChainOutcomeCommentPrompts,
    },
    "csqa": {
        "outcome": ChainOutcomeCommentPrompts,
    },
    "aqua": {
        "outcome": ChainOutcomeCommentPrompts,
    },
    "svamp": {
        "outcome": ChainOutcomeCommentPrompts,
    },
}


def get_system_prompts(data_config: dict):
    """Get the system prompts for the dataset."""
    dataset_name = data_config["data_name"].lower()
    logging.info("Get system prompts for %s.", dataset_name)
    return system_prompts[dataset_name]


def get_thought_prompts(data_config: dict):
    """Get the thought prompts for the dataset."""
    dataset_name = data_config["data_name"].lower()
    logging.info("Get thought prompts for %s.", dataset_name)
    return thought_prompts[dataset_name]


def get_chain_comment_prompts(data_config: dict):
    """Get the chain comment prompts for the dataset."""
    dataset_name = data_config["data_name"].lower()
    logging.info("Get chain comment prompts for %s.", dataset_name)
    return chain_comment_prompts[dataset_name]
