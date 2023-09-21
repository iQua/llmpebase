"""
The implementation of our Federation of Thought.
"""

import os
import resource

from vgbase.utils.envs_utils import define_env
from vgbase.config import Config
from dotenv import load_dotenv

from llmpebase.models.LMs import chatgpts

# from llmpebase.datasets.mmlu import DataSource as mmlu_datasource
from llmpebase.datasets.mmlu import (
    DataSource as mmlu_datasource,
)
from llmpebase.models.LMs_prompting import mmlu_prompt


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

    #################### Prepare dataset ####################
    # build data module
    # for train
    mmlu_data = mmlu_datasource()
    train_set = mmlu_data.get_train_set()
    test_set = mmlu_data.get_test_set()

    model_config = Config.items_to_dict(model_config._asdict())
    chatgpt_api = chatgpts.ChatGPTAPIRequest(model_config, env_config)
    chatgpt_api.get_authorization(
        organization=os.getenv("OPENAI_ORGAN_KEY"), api_key=os.getenv("OPENAI_API_KEY")
    )

    task_name = "abstract algebra"
    task_samples = train_set[task_name, -1]
    test_sample = test_set[task_name, 0]

    input_prompt = mmlu_prompt.MMLUStandardPrompt()
    # input_prompt = mmlu_prompt.MMLUCoTPrompt(
    #     cot_filepath="examples/LMs/ChainOfThought/mmlu-cot-claude.json"
    # )
    request_prompt = input_prompt.organize_test_prompt(
        task_name, task_samples, test_sample
    )
    print(request_prompt)

    do_model_request(chatgpt_api, request_prompt)


if __name__ == "__main__":
    _main()
