"""
Implementation of Chain Of Thought [1].

[1]. Wei, et.al., Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, 23.
"""

from llmpebase.config import Config

from llmpebase.models import define_model, define_prompt
from llmpebase.datasets import define_dataset
from llmpebase.utils import recorder


def do_model_request(model, request_prompt):
    """Do model request."""
    ipt_msg = model.create_format_input(
        user_prompt=request_prompt,
        sys_prompt=f"""Follow the given examples and answer the question. Please utilize a sub-sentence '{model.target_answer_format}' to summarize the core response/answer/result for users to read.""",
    )
    model_responses = model.perform_request(
        input_request=ipt_msg, per_request_responses=2
    )

    extracted_contents = model.extract_response_contents(model_responses)

    return extracted_contents


def perform_eval(model, train_set, test_set, input_prompter, logging_config, config):
    """Performing the evaluation."""
    eval_recorder = recorder.DualExtensionRecoder(
        records_filename="records",
        samples_filename="samples",
        record_path=logging_config.result_path,
        record_name="llm_records",
        is_append=True,
    )

    eval_recorder.set_check_items(
        sample_check_item="answer", record_check_item="request_prompt"
    )

    for prompt, eval_sample, eval_groundtruth in input_prompter.evaluater(
        train_set, test_set, config
    ):
        contents = do_model_request(model, request_prompt=prompt)
        extracted_target_answers = input_prompter.extract_target_answers(contents)
        extracted_target_results = [
            input_prompter.extract_target_result(dst_answer)
            for dst_answer in extracted_target_answers
        ]
        consistency = [
            input_prompter.measure_answers(
                dst_answer=eval_groundtruth, src_answer=dst_answer
            )
            for dst_answer in extracted_target_answers
        ]

        record = {
            "request_prompt": prompt,
            "responses": contents,
            "extracted_answers": extracted_target_answers,
            "extracted_results": extracted_target_results,
            "is_correct_answers": consistency,
        }

        eval_recorder.add_one_record(
            sample=eval_sample, record=record, sample_id="question"
        )
        eval_recorder.save_records()


def _main():
    """The core function for model running."""

    # obtain configs
    model_config = Config().model
    data_config = Config().data
    eval_config = Config().evaluation
    logging_config = Config().logging

    model_config = Config.items_to_dict(model_config._asdict())
    data_config = Config.items_to_dict(data_config._asdict())
    eval_config = Config.items_to_dict(eval_config._asdict())

    #################### Prepare model ####################

    request_model = define_model(model_config)

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
        logging_config=logging_config,
        config=model_config,
    )


if __name__ == "__main__":
    _main()
