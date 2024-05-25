"""
A running pipeline for the plan optimization phase of p-RAG.
"""

import random
import logging

from llmpebase.pipeline import Pipeline


class PlanRetrievalPipeline(Pipeline):
    """
    A pipeline to accumulate the experiences for the SnowBall.
    """

    def __init__(self, reasoner, *args):
        super().__init__(reasoner, *args)

        self.retrieval_config = self.model_config["retrieval"]

    def execute(self):
        """Execute the pipeline to obtain the results."""

        logging.info("Testing on %d samples.", len(self.testset))

        for idx, sample in enumerate(self.testset):

            sample_info = sample["auxiliary"]["sample_info"]
            sample_id = sample_info["sample_id"]
            record_name = f"Idx {idx}-ID<{sample_id}>"
            if record_name in self.exist_records:
                continue
            # Forward the reasoning
            prompt_sample, groundtruth = self.data_prompter.create_prompt_sample(
                sample=sample,
                dataset=self.trainset,
                config=self.model_config,
            )

            contents = self.reasoner.forward(
                prompt_sample=prompt_sample,
                sample_name=record_name,
                sample_info=sample_info,
            )
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
            measurements = self.evaluator.forward(results, groundtruths)

            output = {
                "request_prompt": str(prompt_sample),
                "responses": contents,
                "groundtruth": groundtruths,
                "results": results,
                "measurements": measurements,
            }

            self.recorder.save_one_record(
                sample=sample,
                output=output,
                statistic=self.reasoner.get_cost_statistics(latest=True),
                sample_name=record_name,
            )

            # Reset the reasoning after processing current sample
            self.reasoner.reset_reasoning()
