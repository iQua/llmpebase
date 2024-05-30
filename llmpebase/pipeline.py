"""
A base pipeline used to perform the whole prompt engineering process for the LLM
on the specific dataset.

This pipeline also works as a demo showing how to combine and utilize different 
components of the `llmpebase`.
"""

from typing import Union
import logging

import torch

from llmpebase.model.LM.base import BaseLlmRequest
from llmpebase.model.prompting.base import BasePrompting
from llmpebase.dataset.base import BaseDataset
from llmpebase.extractor.base import BaseReExtractor, BaseLlmExtractor
from llmpebase.evaluator.base import BaseEvaluator, BaseLLMEvaluator

from llmpebase.model import define_prompt, define_model
from llmpebase.dataset import define_dataset
from llmpebase.extractor import get as get_extractor
from llmpebase.evaluator import get as get_evaluator

from llmpebase.utils import recorder
from llmpebase.config import Config


class Pipeline:
    """A base pipeline to perform the prompt engineering.

    Args:
        reasoner: A torch.nn.Module to perform the reasoning toward
         answering the question.
         dataset: The dataset containing the Q&A pairs.
        prompter: A data-specific prompter to generate the prompt for the
         reasoning process.
        extractor: The extractor used to extract the target result from the response
            of the LLM.
    """

    def __init__(
        self,
        reasoner: Union[BaseLlmRequest, torch.nn.Module],
        dataset: BaseDataset = None,
        data_prompter: BasePrompting = None,
        extractor: Union[BaseReExtractor, BaseLlmExtractor] = None,
        evaluator: Union[BaseEvaluator, BaseLLMEvaluator] = None,
    ):
        # The LLM model used to perform the request.
        self.reasoner = reasoner
        # The dataset
        self.dataset = dataset
        # The prompting used to generate the prompt
        self.data_prompter = data_prompter
        # The extractor used to extract the target result from the response
        # of the LLM
        self.extractor = extractor
        # The evaluator used to evaluate the result
        self.evaluator = evaluator

        # Basic components of the pipeline
        # ID of the pipeline
        self.pipeline_id: int = 0

        # The flag to resume the pipeline
        self.resume = True
        # The latest sample's index should be used by the pipeline
        # to perform reasoning
        # This is used for the resume
        self.exist_records = []

        # The train, test, and val datasets
        self.trainset = None
        self.testset = None
        self.valset = None

        # Record the results
        self.recorder = None

        # Get the configuration
        self.model_config = Config.items_to_dict(Config().model._asdict())
        self.data_config = Config.items_to_dict(Config().data._asdict())
        self.log_config = Config.items_to_dict(Config().logging._asdict())

    def setup(self):
        """Configuration of the pipeline."""
        eval_config = Config.items_to_dict(Config().evaluation._asdict())

        self.resume = eval_config["do_resume"] if "do_resume" in eval_config else True

        if self.data_prompter is None:
            self.data_prompter = define_prompt(
                self.data_config,
                self.model_config,
            )

        if self.extractor is None:
            config = eval_config["extractor"]
            extractor = get_extractor(
                data_name=self.data_config["data_name"],
                config=config,
            )
            if config["style"] == "llm":
                if "llm_config" not in config:
                    config["llm_config"] = self.model_config
                self.extractor = extractor(llm_model=define_model(config["llm_config"]))

        if self.evaluator is None:
            self.evaluator = get_evaluator(
                data_name=self.data_config["data_name"], style=eval_config["style"]
            )()

        if self.recorder is None:
            self.recorder = recorder.BaseRecorder(
                output_filename="outputs",
                sample_filename="samples",
                record_path=self.log_config["result_path"],
                record_name="llm_records",
            )

        # Resume the pipeline
        if self.resume:
            self.exist_records = self.recorder.get_recorded_names()
            logging.info(
                "[Pipeline %d] Resume from #sample %d.",
                self.pipeline_id,
                len(self.exist_records),
            )

    def load_data(self):
        """Load the datasets for the pipeline."""

        if self.dataset is None:
            self.dataset = define_dataset(self.data_config)

        if self.trainset is None:
            self.trainset = self.dataset.get_train_set()
        if self.testset is None:
            self.testset = self.dataset.get_test_set()

    def execute(self):
        """Execute the pipeline to obtain the results."""

        for idx, sample in enumerate(self.testset):
            sample_info = sample["auxiliary"]["sample_info"]
            sample_id = sample_info["sample_id"]
            record_name = f"{idx}-ID<{sample_id}>"
            if record_name in self.exist_records:
                continue
            prompt_sample, groundtruth = self.data_prompter.create_prompt_sample(
                sample, self.trainset, self.model_config
            )

            contents = self.reasoner.forward(
                prompt_sample, sample_name=record_name, sample_info=sample_info
            )
            assert isinstance(contents, list)

            results = [
                self.extractor.forward(
                    content,
                    solution_flag=self.data_prompter.solution_flag,
                    problem_name=sample_info["sample_problem"],
                    question=sample["question"],
                )
                for content in contents
            ]
            groundtruths = [groundtruth] * len(results)
            comparison = self.evaluator.forward(results, groundtruths)
            output = {
                "request_prompt": str(prompt_sample),
                "responses": contents,
                "groundtruth": groundtruths,
                "results": results,
                "comparison": comparison,
            }

            self.recorder.save_one_record(
                sample=sample,
                output=output,
                statistic=self.reasoner.get_cost_statistics(latest=True),
                sample_name=record_name,
            )
            # Reset the reasoning after processing current sample
            self.reasoner.reset_reasoning()
