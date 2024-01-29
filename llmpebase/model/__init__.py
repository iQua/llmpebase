"""
An interface of models and prompts
"""
import logging

from llmpebase.model.LM import gpts, llama2_hf, llama2_meta
from llmpebase.model.prompting import (
    base,
    zeroshot_cot,
    cot,
    fewshot,
)

llms_factory = {
    "gpt": gpts.GPTAPIRequest,
    "meta-llama": llama2_meta.LLaMARequest,
    "llama2_hf": llama2_hf.llama2Request,
}


prompts_factory = {
    "mmlu": {
        "fewshot": fewshot.MMLUFewShotPrompting,
        "cot": cot.MMLUCoTPrompting,
        "zeroshot_cot": zeroshot_cot.MMLUZeroShotCoTPrompting,
    },
    "gsm8k": {
        "fewshot": base.BasePrompting,
        "cot": cot.GSM8KCoTPrompting,
        "zeroshot_cot": base.BaseZeroShotCoTPrompting,
    },
    "gameof24": {
        "fewshot": "Not provided",
        "cot": "Not provided",
        "zeroshot_cot": zeroshot_cot.GameOf24ZeroShotCoTPrompting,
    },
    "math": {
        "fewshot": fewshot.ProblemFewShotPrompting,
        "cot": cot.MATHCoTPrompting,
        "zeroshot_cot": base.BaseZeroShotCoTPrompting,
    },
    "bbh": {
        "fewshot": fewshot.ProblemFewShotPrompting,
        "cot": cot.BBHCoTPrompting,
        "zeroshot_cot": base.BaseZeroShotCoTPrompting,
    },
    "theoremqa": {
        "fewshot": fewshot.TheoremQAFewShotPrompting,
        "cot": cot.TheoremQACoTPrompting,
        "zeroshot_cot": zeroshot_cot.TheoremQAZeroShotCoTPrompting,
    },
    "csqa": {
        "zeroshot_cot": zeroshot_cot.CSQAZeroShotCoTPrompting,
    },
    "aqua": {
        "fewshot": fewshot.AQUAFewShotPrompting,
        "zeroshot_cot": zeroshot_cot.AQUAZeroShotCoTPrompting,
        # CoT of AQUA is the same as the fewshot as the
        # 'rationale' is provided by the dataset
        "cot": fewshot.AQUAFewShotPrompting,
    },
    "svamp": {
        "fewshot": fewshot.ProblemFewShotPrompting,
        "zeroshot_cot": base.BaseZeroShotCoTPrompting,
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
