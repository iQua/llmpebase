"""
The implementation of the BoT.
"""

import BoT_reasoner
import BoT_commenter
import BoT_core


from llmpebase.config import Config
from llmpebase.models import define_model, define_prompt
from llmpebase.datasets import define_dataset
from llmpebase.utils import recorder


def perform_bot_eval(
    bot_model, test_set, data_prompter, logging_config: dict, model_config: dict
):
    """Performing the evaluation with BoT."""
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

    # Remove the notice, which is not suitable for BoT
    data_prompter.notice = ""
    # Visit all test samples for evaluation
    for task_prompt, eval_sample, eval_groundtruth in data_prompter.evaluater(
        train_set=None, eval_set=test_set, config=None
    ):
        best_reasoning_chain = bot_model.perform_bot_reasoning(task_prompt=task_prompt)
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

    # Define the BoT model
    bot_model = BoT_core.BoostOfThoughts(
        experience_reasoner=experience_reasoner,
        chain_commenter=chain_commenter,
        model_config=model_config,
    )

    perform_bot_eval(
        bot_model,
        test_set=test_set,
        data_prompter=prompter,
        logging_config=logging_config,
        model_config=model_config,
    )


if __name__ == "__main__":
    _main()
