"""
Implementation of the federated running structure.
"""

from typing import List, Dict

import numpy as np


from llmpebase.models.LMs_prompting import residual_tree_of_thoughts
from llmpebase.models.LMs import chatgpts

import fot_model
import chains_aggregation


def perform_user_operation(
    user_id: int, prompts: dict, local_update_config: dict, round_idx: int
):
    """Perform the operation of the user - the residual tree structure
    will be built.

    :param prompts: A `dict` containing three terms,
     - initial_prompt: contains the question and some description
     - global_prompt: A string showing the reasoning chain and the corresponing
      score.
    """

    initial_prompt = prompts["initial_prompt"]
    global_prompt = prompts["global_prompt"]

    # in the first, there is no residual thought
    # we, therefore, set the global prompt to be None
    # to avoid the residual thought during tree construction
    if round_idx == 0:
        global_prompt = None

    model_config = local_update_config["model_config"]
    envs_config = local_update_config["envs_config"]
    base_request_model = chatgpts.ChatGPTAPIRequest(model_config, envs_config)

    thought_model = fot_model.ThoughtModel(base_request_model)
    tree_builder = residual_tree_of_thoughts.RToTLevelWiseBest(
        thought_model, n_child_nodes=2
    )
    tree_builder.construct_tree_root(
        though=initial_prompt, residual_though=global_prompt
    )

    # perform the reasoning with the tree structure
    tree_builder.build_thought_tree()

    # extracting the best reasoning chain from the tree
    thought_chain = tree_builder.get_best_though_chain()

    # organizing the best chain to be a prompt
    local_prompt = thought_model.organize_thoughs_chain_prompt(thought_chain)
    return {user_id: {"local_prompt": local_prompt, "thought_chain": thought_chain}}


def perform_server_operation(global_prompt: str, users_update: List[Dict[int, dict]]):
    """Performing the operation of the server - tree branches updated
    by users will be aggregated."""

    users_chain = {
        update["user_id"]: update["thought_chain"] for update in users_update
    }

    aggregated_chain = chains_aggregation.leaf_depend_aggregation(chains=users_chain)

    thought_model = fot_model.ThoughtModel(None)
    global_prompt = thought_model.organize_thoughs_chain_prompt(aggregated_chain)

    return global_prompt


def do_fot_reasoning(initial_prompt: str, fot_config: dict):
    """Do the FoT reasoning for the question solving."""
    num_rounds = fot_config["num_rounds"]
    num_users = fot_config["num_users"]
    num_users_per_round = fot_config["num_users_per_round"]

    local_update_config = fot_config["local_update"]

    # create users id
    users_id = list(range(1, num_users + 1))

    # setting the global prompt to be the initial prompt

    shared_prompts = {"initial_prompt": initial_prompt, "global_prompt": initial_prompt}

    for round_idx in range(num_rounds):
        users_update = []
        # select random clients
        selected_users = np.random.choice(users_id, num_users_per_round)

        # client update
        for user_id in selected_users:
            local_update = perform_user_operation(
                user_id, shared_prompts, local_update_config, round_idx
            )
            users_update.append(local_update)
        # serer aggregate
        global_prompt = perform_server_operation(global_prompt, users_update)
        # update the global prompt
        shared_prompts["global_prompt"] = global_prompt
