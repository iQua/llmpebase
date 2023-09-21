"""
The implementation of Chain Of Thought [1].

[1]. Wei, et.al., Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, 23.
"""

import os
import random
import resource


from vgbase.utils.envs_utils import define_env
from vgbase.config import Config
from dotenv import load_dotenv

from llmpebase.models.LMs import chatgpts

# from llmpebase.datasets.mmlu import DataSource as mmlu_datasource
from llmpebase.datasets.mmlu import (
    DataSource as mmlu_datasource,
)
from llmpebase.datasets.gsm8k import (
    DataSource as gsm8k_datasource,
)
from llmpebase.models.LMs_prompting import mmlu_prompt
from llmpebase.models.LMs_prompting import gsm8k_prompt

# there must have a .env file containing keywords
# OPENAI_ORGAN_KEY and OPENAI_API_KEY
load_dotenv()


def do_model_request(chatgpt_api, request_prompt):
    """Do model request."""
    ipt_msg = chatgpt_api.create_format_input(
        user_prompt=request_prompt,
        sys_prompt="Follow the given examples and answer the question.",
    )
    model_responses = chatgpt_api.perform_request(
        request_input=ipt_msg, per_request_responses=3
    )
    print("model_responses: ", model_responses)

    extract_answer = chatgpt_api.extract_answers(model_responses)
    print(extract_answer)
    extract_target_answer = chatgpt_api.extract_response_target_answer(extract_answer)
    print(extract_target_answer)


def eval_mmlu(chatgpt_api, eval_config):
    """Eval the MMLU."""
    mmlu_data = mmlu_datasource()
    train_set = mmlu_data.get_train_set()
    test_set = mmlu_data.get_test_set()

    input_prompter = mmlu_prompt.MMLUStandardPrompt()
    # input_prompt = mmlu_prompt.MMLUCoTPrompt(
    #     cot_filepath="examples/LMs/ChainOfThought/mmlu-cot-claude.json"
    # )
    chatgpt_api.set_target_answer_format(input_prompter.answer_format_str)

    n_shots = eval_config["n_shots"]

    print("n_shots: ", n_shots)
    assert n_shots <= 5
    for task_name in train_set.tasks_name:
        train_samples = train_set[task_name, -1]
        test_task_samples = test_set[task_name, -1]
        shots = train_samples[:n_shots]
        for test_sample in test_task_samples:
            request_prompt = input_prompter.organize_test_prompt(
                task_name, shots, test_sample
            )
            do_model_request(chatgpt_api, request_prompt)


def eval_gsm8k(chatgpt_api, eval_config):
    """Eval the GSM8K."""
    gsm_data = gsm8k_datasource()
    train_set = gsm_data.get_train_set()
    test_set = gsm_data.get_test_set()

    # input_prompt = gsm8k_prompt.GSM8KStandardPrompt()
    input_prompter = gsm8k_prompt.GSM8KCoTPrompt(
        cot_filepath="examples/ChainOfThought/gsm8k_prompt/prompt_6_9step.txt"
    )
    chatgpt_api.set_target_answer_format(input_prompter.answer_format_str)

    n_shots = eval_config["n_shots"]

    n_test_samples = len(test_set)
    print("n_test_samples: ", n_test_samples)
    for test_idx, test_sample in enumerate(test_set):
        samples = [train_set[random.randint(0, len(test_set))] for _ in range(n_shots)]
        request_prompt = input_prompter.organize_test_prompt(
            task_name=None, few_shot_samples=samples, test_sample=test_sample
        )
        do_model_request(chatgpt_api, request_prompt)


datasets_eval = {"GSM8K": eval_gsm8k, "MMLU": eval_mmlu}


def _main():
    """The core function for model running."""

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # obtain configs
    model_config = Config().model
    data_config = Config().data
    train_config = Config().train
    eval_config = Config().evaluation
    logging_config = Config().logging

    env_config = Config().environment
    env_config = Config().items_to_dict(env_config._asdict())

    # define the environment used for learning
    devices, env_config = define_env(env_config=env_config)

    #################### Prepare model ####################
    model_config = Config.items_to_dict(model_config._asdict())
    chatgpt_api = chatgpts.ChatGPTAPIRequest(model_config, env_config)
    chatgpt_api.get_authorization(
        organization=os.getenv("OPENAI_ORGAN_KEY"), api_key=os.getenv("OPENAI_API_KEY")
    )

    #################### Do evaluation for the dataset ####################
    eval_config = Config.items_to_dict(eval_config._asdict())
    datasets_eval[data_config.data_name](chatgpt_api, eval_config)


if __name__ == "__main__":
    _main()
