"""
An interface of models and prompts
"""
import logging

from llmpebase.model.LM import gpts, llama_falcon, llama_pipeline  # , llama2
from llmpebase.model.prompting import bbh, game24, gsm8k, math, mmlu, theoremqa

llms_factory = {
    "gpt": gpts.GPTAPIRequest,
    "llama": llama_falcon.LLaMARequest,
    "llama_pipeline": llama_pipeline.LLaMAPipelineRequest,
    # "llama2": llama2.llama2Request,
}


prompts_factory = {
    "mmlu": {
        "standard": mmlu.MMLUStandardPrompting,
        "cot": mmlu.MMLUCoTPrompting,
        "zeroshot_cot": mmlu.MMLUZeroShotCoTPrompting,
    },
    "gsm8k": {
        "standard": gsm8k.GSM8KStandardPrompting,
        "cot": gsm8k.GSM8KCoTPrompting,
        "zeroshot_cot": gsm8k.GSM8KZeroShotCoTPrompting,
    },
    "gameof24": {
        "standard": "Not provided",
        "cot": "Not provided",
        "zeroshot_cot": game24.GameOf24ZeroShotPrompting,
    },
    "math": {
        "standard": math.MATHStandardPrompting,
        "cot": math.MATHCoTPrompting,
        "zeroshot_cot": math.MATHZeroShotCoTPrompting,
    },
    "bbh": {
        "standard": bbh.BBHStandardPrompting,
        "cot": bbh.BBHCoTPrompting,
        "zeroshot_cot": bbh.BBHZeroShotCoTPrompting,
    },
    "theoremqa": {
        "standard": theoremqa.TheoremQAStandardPrompting,
        "cot": theoremqa.TheoremQACoTPrompting,
        "zeroshot_cot": theoremqa.TheoremQAZeroShotCoTPrompting,
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