"""
An interface of models and prompts
"""


from llmpebase.models.LMs import gpts, llama_falcon, llama_pipeline, llamav2
from llmpebase.models.prompting import gsm8k_prompting, mmlu_prompting, game24_prompting

models_factory = {
    "gpt": gpts.GPTAPIRequest,
    "llama": llama_falcon.LLaMARequest,
    "llama_pipeline": llama_pipeline.LLaMAPipelineRequest,
    "llamav2": llamav2.LLaMAV2Request,
}


prompts_factory = {
    "mmlu": {
        "standard": mmlu_prompting.MMLUStandardPrompting,
        "cot": mmlu_prompting.MMLUCoTPrompting,
        "zeroshot_cot": mmlu_prompting.MMLUZeroShotCoTPrompting,
    },
    "gsm8k": {
        "standard": gsm8k_prompting.GSM8KStandardPrompting,
        "cot": gsm8k_prompting.GSM8KCoTPrompting,
        "zeroshot_cot": gsm8k_prompting.GSM8KZeroShotCoTPrompting,
    },
    "gameof24": {
        "standard": game24_prompting.GameOf24StandardPrompting,
        "cot": game24_prompting.GameOf24StandardPrompting,
        "zeroshot_cot": game24_prompting.GameOf24StandardPrompting,
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
