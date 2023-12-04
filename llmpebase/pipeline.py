"""
A base pipeline used to perform the whole prompt engineering process for the LLM
on the specific dataset.

This pipeline also works as a demo showing how to combine and utilize different 
components of the `llmpebase`.
"""

from typing import Union, List

import torch

from llmpebase.model.LM.base import BaseLMRequest
from llmpebase.dataset.base import BaseDataset
from llmpebase.extractor.base import BaseReExtractor, BaseLLMExtractor
from llmpebase.model.prompting.base import BasePrompting
from llmpebase.evaluator.base import BaseEvaluator, BaseLLMEvaluator
from llmpebase.model import define_model, define_prompt
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
        reasoner: Union[BaseLMRequest, torch.nn.Module],
        dataset: BaseDataset,
        prompter: BasePrompting,
        extractor: Union[BaseReExtractor, BaseLLMExtractor],
        evaluator: Union[BaseEvaluator, BaseLLMEvaluator] = None,
    ):
        # The LLM model used to perform the request.
        self.reasoner = reasoner
        # The dataset
        self.dataset = dataset
        # The prompting used to generate the prompt
        self.prompter = prompter
        # The extractor used to extract the target result from the response
        # of the LLM
        self.extractor = extractor
        # The evaluator used to evaluate the result
        self.evaluator = evaluator

        # Basic componenet for the pipeline
        # ID of the pipeline
        self.pipeline_id: int = 0
        # sampler used by the dataset of this pipeline
        self.sampler = None

        # The train, test, and val datasets
        self.trainset = None
        self.testset = None
        self.valset = None

        # Record the results
        self.recorder = None

        # Get the configuration
        self.model_config = Config.items_to_dict(Config().model._asdict())
        self.eval_config = Config.items_to_dict(Config().evaluation._asdict())
        self.data_config = Config.items_to_dict(Config().data._asdict())

    def configuration(self):
        """Configuration of the pipeline."""

        eval_config = Config().evaluation
        logging_config = Config().logging

        if self.reasoner is None:
            # If no customized model is provided, use the
            # LLM model defined in the config
            self.reasoner = define_model(self.model_config)
        else:
            self.reasoner = self.reasoner(self.model_config)

        if self.prompter is None:
            self.prompter = define_prompt(
                self.data_config,
                self.model_config,
            )

        if self.extractor is None:
            self.extractor = get_extractor(
                data_name=self.data_config["name"],
                style=eval_config["extractor"]["style"],
            )()

        if self.evaluator is None:
            self.evaluator = get_evaluator(
                data_name=self.data_config["name"], style=eval_config["style"]
            )()

        # Set the target answer format
        self.reasoner.set_solution_flag(self.prompter.solution_str)

        if self.recorder is None:
            self.recorder = recorder.DualExtensionRecoder(
                records_filename="records",
                samples_filename="samples",
                record_path=logging_config.result_path,
                record_name="llm_records",
                is_append=True,
            )

            self.recorder.set_check_items(
                sample_check_item="answer", record_check_item="request_prompt"
            )

    def load_data(self):
        """Load the datasets for the pipeline."""

        if self.dataset is None:
            self.dataset = define_dataset(self.data_config)

        if self.trainset is None:
            self.trainset = self.dataset.get_train_set()
        if self.testset is None:
            self.testset = self.dataset.get_test_set()

    def perform_reasoning(self, request_prompt: str) -> List[str]:
        """Perform the reasoning process.

        :return contents: The contents of the responses from reasoner.
        """
        return self.reasoner.forward(request_prompt)

    def get_requests(self):
        """Get the number of requests performed during reasoning."""
        return self.reasoner.num_requests

    def forward(self):
        """Forward the pipeline to obtain the results."""

        for sample in self.testset:
            prompt_sample, groundtruth = self.prompter.create_prompt_sample(
                sample, self.trainset, self.eval_config
            )

            contents = self.perform_reasoning(request_prompt=prompt_sample)
            results = [self.extractor(content) for content in contents]
            groundtruths = [groundtruth] * len(results)
            comparsion = self.evaluator.forward(results, groundtruths)
            record = {
                "request_prompt": prompt_sample,
                "responses": contents,
                "groundtruth": results,
                "results": groundtruths,
                "comparsion": comparsion,
            }

            self.recorder.add_one_record(
                sample=sample, record=record, sample_id="question"
            )
            self.recorder.save_records()
