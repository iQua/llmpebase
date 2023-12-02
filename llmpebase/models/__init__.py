"""
An interface of models and prompts
"""


from llmpebase.models.LMs import gpts, llama_falcon, llama_pipeline, llamav2
from llmpebase.models.prompting import bbh, game24, gsm8k, math, mmlu, theoremqa

models_factory = {
    "gpt": gpts.GPTAPIRequest,
    "llama": llama_falcon.LLaMARequest,
    "llama_pipeline": llama_pipeline.LLaMAPipelineRequest,
    "llamav2": llamav2.LLaMAV2Request,
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
        "standard": game24.GameOf24StandardPrompting,
        "cot": game24.GameOf24StandardPrompting,
        "zeroshot_cot": game24.GameOf24StandardPrompting,
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

    return models_factory[model_type](model_config)


def define_prompt(data_config: dict, model_config: dict):
    """Define the datasets based on the config file."""
    data_name = data_config["data_name"].lower()
    prompt_type = model_config["prompt_type"].lower()

    return prompts_factory[data_name][prompt_type](model_config)
