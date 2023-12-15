"""
An interface of models and prompts
"""
import logging

from llmpebase.model.LM import gpts, llama_falcon, llama_pipeline  # , llama2
from llmpebase.model.prompting import (
    base,
    zeroshot,
    cot,
    fewshot,
)

llms_factory = {
    "gpt": gpts.GPTAPIRequest,
    "llama": llama_falcon.LLaMARequest,
    "llama_pipeline": llama_pipeline.LLaMAPipelineRequest,
    # "llama2": llama2.llama2Request,
}


prompts_factory = {
    "mmlu": {
        "fewshot": fewshot.MMLUFewShotPrompting,
        "cot": cot.MMLUCoTPrompting,
        "zeroshot": zeroshot.MMLUZeroShotPrompting,
    },
    "gsm8k": {
        "standard": base.BasePrompting,
        "cot": cot.GSM8KCoTPrompting,
        "zeroshot": base.BaseZeroShotPrompting,
    },
    "gameof24": {
        "standard": "Not provided",
        "cot": "Not provided",
        "zeroshot": zeroshot.GameOf24ZeroShotPrompting,
    },
    "math": {
        "standard": fewshot.MATHFewShotPrompting,
        "cot": cot.MATHCoTPrompting,
        "zeroshot": base.BaseZeroShotPrompting,
    },
    "bbh": {
        "standard": fewshot.BBHFewShotPrompting,
        "cot": cot.BBHCoTPrompting,
        "zeroshot": base.BaseZeroShotPrompting,
    },
    "theoremqa": {
        "standard": fewshot.TheoremQAFewShotPrompting,
        "cot": cot.TheoremQACoTPrompting,
        "zeroshot": zeroshot.TheoremQAZeroShotPrompting,
    },
}


def define_model(model_config: dict):
    """Define the datasets based on the config file."""
    model_type = model_config["model_type"].lower()
    llm_model = llms_factory[model_type](model_config)
    llm_model.configuration()

    logging.info("Defined LLM model %s.", model_type)
    return llm_model


def define_prompt(data_config: dict, model_config: dict):
    """Define the datasets based on the config file."""
    data_name = data_config["data_name"].lower()
    prompt_type = model_config["prompt_type"].lower()
    logging.info("Defined %s Prompting for %s .", prompt_type, data_name)

    return prompts_factory[data_name][prompt_type](model_config)
