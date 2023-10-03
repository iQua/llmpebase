"""
The implementation of Game of 24.
"""

import os
import resource

from vgbase.utils.envs_utils import define_env
from vgbase.config import Config
from dotenv import load_dotenv


from llmpebase.models.LMs import llamav2
from llmpebase.models.LMs import chatgpts


import bot_model
from llmpebase.models.LMs_prompting import residual_tree_of_thoughts


# there must have a .env file containing keywords
# OPENAI_ORGAN_KEY and OPENAI_API_KEY
load_dotenv()


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

    if "gpt" in model_config["model_name"]:
        # this is the chatgpt model
        request_model = chatgpts.ChatGPTAPIRequest(model_config, env_config)
        request_model.get_authorization(
            organization=os.getenv("OPENAI_ORGAN_KEY"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    request_model.set_target_answer_format(
        solution_format="In summary, the two numbers are:, the arithmetic operation is :, and the new number set is: ."
    )

    task_instruction = "In each step: First, you randomly select two numbers from the current number set to perform Addition, subtraction, multiplication, or division to obtain a new number. Second, you should delete the selected two numbers from your current set. Then, after deleting, if there is no remaining number after deleting, just use the obtained new number as a set. Otherwise, you combine the remaining numbers and the obtained new number into a new set for subsequent usage."

    gen_model = bot_model.ThoughtModel(
        request_model=request_model,
    )
    gob_tree = residual_tree_of_thoughts.RToTLevelWise(
        gen_model,
        n_child_nodes=2,
        model_config=model_config,
    )
    gob_tree.construct_tree_root(
        thought=f"""This is a mathematical reasoning challenge, where the goal is to use four given numbers and basic arithmetic operations, including addition, subtraction, multiplication, and division, to obtain 24. {task_instruction}.
        The given four numbers are: 2, 4, 5, 5""",
        thought_score=None,
    )
    print("gob_tree.max_steps: ", gob_tree.max_steps)
    gob_tree.build_thought_tree()

    gob_tree.print_tree_structure()

    leveas_node = gob_tree.root.leaves
    best_value = 0
    best_leaf = None
    for node in leveas_node:
        node_score = node.thought_score
        if node_score > best_value:
            best_value = node_score
            best_leaf = node

    # For feedback purpose
    feedback = gen_model.get_thought_chain_feedback(thoughts_node_chain=best_leaf.path)
    print(feedback)


if __name__ == "__main__":
    _main()
