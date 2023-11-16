"""
The implementation of applying BoT with one single tree and single reasoning process
on the Game of 24.

This variant of BoT can be compated with the tree of thoughts [1], which performs the 
reasoning by building thoughts in a tree structure.

[1]. Tree of Thoughts: Deliberate Problem Solving with Large Language Models, 2023.
"""

import BoT_reasoner
import BoT_commenter

from llmpebase.config import Config
from llmpebase.models import define_model, define_prompt
from llmpebase.datasets import define_dataset
from llmpebase.utils import recorder

from llmpebase.models.prompting import tree_thoughts


def perform_bot_reasoning(
    task_prompt,
    experience_reasoner,
    chain_commenter,
    model_config,
):
    """Perform the reasoning by using BoT with single tree and single iteration."""
    gob_tree = tree_thoughts.RTTLevelWise(
        experience_reasoner,
        n_child_nodes=model_config["tree_settings"]["n_child_nodes"],
        model_config=model_config,
    )

    # Add the initial task prompt to the root node
    gob_tree.construct_tree_root(
        thought=task_prompt,
        thought_score=None,
    )
    # Build the thought tree to perform the reasoning
    gob_tree.build_thought_tree()
    gob_tree.print_tree_structure()

    best_chain, _ = gob_tree.get_best_thought_chain()

    print("best_chain:", best_chain)

    # Convert the chain to the prompt
    chain_prompt = experience_reasoner.organize_though_chain_prompt(
        node_thought_chain=best_chain
    )
    print("chain_prompt:", chain_prompt)

    # Get the feedback from the commenter
    comment_feedback, chain_prompt = chain_commenter.get_thought_chain_feedback(
        task_prompt=task_prompt, reasoning_chain_content=chain_prompt
    )

    experience = experience_reasoner.create_experience(comment_feedback, chain_prompt)
    experience_reasoner.memory_experience(experience)

    return chain_prompt


def perform_bot_eval(
    experience_reasoner,
    chain_commenter,
    test_set,
    input_prompter,
    logging_config: dict,
    model_config: dict,
):
    """Perform the boosting of reasoning with BoT."""
    eval_recorder = recorder.DualExtensionRecoder(
        records_filename="records",
        samples_filename="samples",
        record_path=logging_config["result_path"],
        record_name="llm_records",
        is_append=True,
    )
    eval_recorder.set_check_items(
        sample_check_item="answer", record_check_item="request_prompt"
    )

    # The initial experience is empty
    obtain_feedback = ""

    # Add the experience to the BoT reasoner
    experience_reasoner.memory_experience(obtain_feedback)

    # Remove the notice, which is not suitable for BoT
    input_prompter.notice = ""
    # Visit all test samples for evaluation
    tract_idx = 0
    for task_prompt, eval_sample, eval_groundtruth in input_prompter.evaluater(
        train_set=None, eval_set=test_set, config=None
    ):
        tract_idx += 1
        if tract_idx > 700:
            best_reasoning_chain = perform_bot_reasoning(
                task_prompt=task_prompt,
                experience_reasoner=experience_reasoner,
                chain_commenter=chain_commenter,
                model_config=model_config,
            )
            print("best_reasoning_chain: ")
            print(best_reasoning_chain)

            print(ok)


def _main():
    # Obtain configs
    model_config = Config().model
    data_config = Config().data
    eval_config = Config().evaluation
    logging_config = Config().logging

    model_config = Config.items_to_dict(model_config._asdict())
    data_config = Config.items_to_dict(data_config._asdict())
    eval_config = Config.items_to_dict(eval_config._asdict())
    logging_config = Config.items_to_dict(logging_config._asdict())

    #################### Prepare model ####################
    request_model = define_model(model_config)

    #################### Prepare the dataset ##############
    datasource = define_dataset(data_config)
    test_set = datasource.get_test_set()

    #################### Prepare the data prompt ##############
    prompter = define_prompt(data_config, model_config)

    request_model.set_target_answer_format(prompter.answer_format_str)

    #################### Define the BoT model ##############
    # The experience should be included in this BoT model
    experience_reasoner = BoT_reasoner.ExperienceRecallReasoner(
        request_model=request_model,
    )

    # The reasoning chain commenter used by the BoT model
    chain_commenter = BoT_commenter.ReasoningChainCommenter(request_model=request_model)

    perform_bot_eval(
        experience_reasoner,
        chain_commenter,
        test_set,
        input_prompter=prompter,
        logging_config=logging_config,
        model_config=model_config,
    )


if __name__ == "__main__":
    _main()
