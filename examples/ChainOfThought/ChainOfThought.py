"""
The implementation of Chain Of Thought [1].

[1]. Wei, et.al., Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, 23.
"""

import resource


from vgbase.utils.envs_utils import define_env
from vgbase.config import Config

from llmpebase.models import define_model, define_prompt
from llmpebase.datasets import define_dataset


def do_model_request(model, request_prompt):
    """Do model request."""
    ipt_msg = model.create_format_input(
        user_prompt=request_prompt,
        sys_prompt="Follow the given examples and answer the question.",
    )
    model_responses = model.perform_request(
        input_request=ipt_msg, per_request_responses=2
    )

    extracted_contents = model.extract_responses_content(model_responses)

    return extracted_contents


def perform_eval(model, train_set, test_set, input_prompter, eval_config):
    """Performing the evaluation."""

    for prompt in input_prompter.evaluater(train_set, test_set, eval_config):
        print(prompt)

        contents = do_model_request(model, request_prompt=prompt)

        extracted_target_answers = input_prompter.extract_contents_target_answer(
            contents
        )
        print(extracted_target_answers)
        print(ok)


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
    model_config = Config.items_to_dict(model_config._asdict())
    data_config = Config.items_to_dict(data_config._asdict())
    eval_config = Config.items_to_dict(eval_config._asdict())

    # define the environment used for learning
    devices, env_config = define_env(env_config=env_config)

    #################### Prepare model ####################

    request_model = define_model(model_config, env_config)

    #################### Do evaluation for the dataset ####################
    datasource = define_dataset(data_config)
    train_set = datasource.get_train_set()
    test_set = datasource.get_test_set()

    prompter = define_prompt(data_config, model_config)

    request_model.set_target_answer_format(prompter.answer_format_str)

    perform_eval(
        request_model,
        train_set,
        test_set,
        input_prompter=prompter,
        eval_config=eval_config,
    )


if __name__ == "__main__":
    _main()
