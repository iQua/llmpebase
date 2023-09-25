"""
The implementation of Game of 24.
"""

import os
import resource

from vgbase.utils.envs_utils import define_env
from vgbase.config import Config
from dotenv import load_dotenv


from llmpebase.models.LMs import llamav2

import bot_model
from llmpebase.models.LMs_prompting import residual_tree_of_thoughts


def _main():
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

    #################### Prepare model ####################
    model_config = Config.items_to_dict(model_config._asdict())

    # define the environment used for learning
    devices, env_config = define_env(env_config=env_config)

    if "model_type" in model_config and "llamav2" == model_config["model_type"]:
        request_model = llamav2.LLaMAV2Request(model_config, env_config)

    request_model.set_target_answer_format(solution_format="The answer is: .")

    root_node = residual_tree_of_thoughts.ThoughtNode(
        name="1",
        thought="This is a mathematical reasoning challenge, where the goal is to use four given numbers and basic arithmetic operations, including Addition, subtraction, multiplication, and division, to obtain 24. In each step, you can only select two from the number set to perform Addition, subtraction, multiplication, or division to obtain a new number, which is combined with the remaining number to get a new number set for the next step. The given four numbers are: 2, 4, 5, 5.",
        thought_score=None,
    )

    gen_model = bot_model.ThoughtModel(request_model=request_model)

    answers = gen_model.generate_thoughts(
        thoughts_node_chain=[root_node], num_thoughts=20
    )
    print(answers)


if __name__ == "__main__":
    _main()
